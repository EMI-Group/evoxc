"""Fail-closed release checks; run from the checkout (verify, dist) or beside dist/ (other commands)."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import tomllib
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

ENV = os.environ
DIST = Path("dist")


class ReleaseError(Exception):
    pass


def require(valid, message):
    if not valid:
        raise ReleaseError(message)


def output(name, value):
    with open(ENV["GITHUB_OUTPUT"], "a") as out:
        print(f"{name}={value}", file=out)


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def oid(ref):
    return git("rev-parse", "--verify", "--end-of-options", ref)


def verify_source():
    event = json.loads(Path(ENV["GITHUB_EVENT_PATH"]).read_text())
    kind, ref, sha = (ENV[k] for k in ("GITHUB_EVENT_NAME", "GITHUB_REF", "GITHUB_SHA"))
    tag = ""
    event_shas = [sha]
    hf = kind == "release"
    if hf:
        tag = event["release"]["tag_name"]
        require(event["action"] == "published" and not event["release"]["draft"], "Not a published HF release")
    elif kind == "push":
        require(event["created"] and not event["deleted"] and event["ref"] == ref, "Not a new tag push")
        tag = ref.removeprefix("refs/tags/")
        event_shas.append(event["after"])
    else:
        require(kind == "workflow_dispatch" and ref == "refs/heads/main", "Rehearse from main only")

    prefix = "evoxc-hf-v" if hf else "v"
    pyproject = Path("evoxc_hf/pyproject.toml" if hf else "pyproject.toml")
    version = tomllib.loads(pyproject.read_text())["project"]["version"]
    require(re.fullmatch(r"[0-9][0-9A-Za-z.+!_-]*", version), f"Invalid release version in {pyproject}")
    tag_oid = ""
    commit = oid("HEAD^{commit}")
    if kind != "workflow_dispatch":
        require(tag == prefix + version and ref == f"refs/tags/{tag}",
                f"Expected exact tag ref refs/tags/{prefix}{version} from {pyproject}")
        tag_oid = oid(ref)
        require(commit == oid(ref + "^{commit}"), "Checked-out commit does not match the current tag")
    for event_sha in event_shas:
        require(re.fullmatch(r"[a-f0-9]{40}", event_sha), "Expected a full event SHA")
        # Depending on the event, GitHub can supply the tag object or the peeled commit.
        require(event_sha in (commit, tag_oid), "Tag or commit SHA does not match the event")
    if not hf:
        require(subprocess.run(["git", "merge-base", "--is-ancestor", commit,
                                "refs/remotes/origin/main"]).returncode == 0,
                "Release commit must be on origin/main")
    for name, value in dict(checkout_sha=commit, tag=tag, tag_oid=tag_oid, version=version).items():
        output(name, value)
    print(f"Verified {tag or 'main rehearsal'} at {commit} ({version})")


def version_of(pyproject):
    with open(pyproject, "rb") as stream:
        return tomllib.load(stream)["project"]["version"]


def top_level(entries):
    return {entry.split("/")[0] for entry in entries}


def wheel_entries(artifact):
    with zipfile.ZipFile(artifact) as archive:
        return archive.namelist()


def sdist_entries(artifact):
    with tarfile.open(artifact) as archive:
        return archive.getnames()


def dist_check(directory=DIST):
    """Assert the four expected archives exist and each ships only its own package.

    Runs from the checkout holding both `pyproject.toml` files; it needs no
    release environment, git state, network access or third-party package.
    """
    directory = Path(directory)
    version = version_of("pyproject.toml")
    hf_version = version_of("evoxc_hf/pyproject.toml")

    # 1. Exactly one platform abi3 wheel for the root `evoxc` project — never `py3-none-any`.
    wheels = sorted(directory.glob(f"evoxc-{version}-*-*.whl"))
    require(len(wheels) == 1, f"{directory}/evoxc-{version}-*-*.whl: expected exactly one platform abi3 "
                              f"evoxc wheel, found {[wheel.name for wheel in wheels]}")
    wheel = wheels[0]
    pure = directory / f"evoxc-{version}-py3-none-any.whl"
    require(not pure.is_file(), f"{pure}: the root `evoxc` project embeds a Rust/PyO3 extension, "
                                "so its wheel must be platform-specific")
    require(re.fullmatch(rf"evoxc-{re.escape(version)}-cp\d+-abi3-[a-z0-9_]+\.whl", wheel.name),
            f"{wheel}: the evoxc wheel must be a per-platform cp<digits>-abi3 wheel "
            "(it embeds a Rust/PyO3 extension)")

    # 2. All four expected archives are present and nothing else is.
    hf_wheel = directory / f"evoxc_hf-{hf_version}-py3-none-any.whl"
    for artifact in (wheel, directory / f"evoxc-{version}.tar.gz", hf_wheel,
                     directory / f"evoxc_hf-{hf_version}.tar.gz"):
        require(artifact.is_file(), f"missing artifact {artifact} — expected four: an evoxc platform "
                                    "wheel + sdist and an evoxc_hf pure-Python wheel + sdist")
    count = sum(1 for path in directory.iterdir() if path.name.endswith((".whl", ".tar.gz")))
    require(count == 4, f"{directory}: expected exactly 4 wheel/sdist artifacts (evoxc + evoxc_hf, a "
                        f"wheel and an sdist each), got {count} file(s)")

    # 3. The `evoxc` wheel ships exactly `evoxc` (including the compiled extension) + its dist-info.
    entries = wheel_entries(wheel)
    tops = top_level(entries)
    bad = sorted(entry for entry in tops if entry.startswith("evoxc_hf"))
    require(not bad, f"{wheel}: ships {bad} — `evoxc_hf` belongs to the SEPARATE distribution "
                     "`evoxc-hf` (`evoxc_hf/pyproject.toml`) and must never be inside the `evoxc` one")
    expected = {"evoxc", f"evoxc-{version}.dist-info"}
    require(tops == expected, f"{wheel}: top-level entries must be exactly {sorted(expected)}, "
                              f"got {sorted(tops)}")
    extension = "evoxc/_evoxc_native.abi3.so"
    require(extension in entries, f"{wheel}: ships no {extension} — maturin must compile the extension "
                                  "from ./native/ and install it in the `evoxc` package")

    # 4. The `evoxc_hf` wheel ships exactly `evoxc_hf` + its dist-info, never `evoxc` or a compiled file.
    entries = wheel_entries(hf_wheel)
    tops = top_level(entries)
    bad = sorted(entry for entry in entries if entry == "evoxc" or entry.startswith("evoxc/"))
    require(not bad, f"{hf_wheel}: ships the `evoxc` package ({bad[:5]}) — `evoxc` is the workspace ROOT "
                     "distribution and a declared runtime dependency, never vendored into `evoxc-hf`")
    expected = {"evoxc_hf", f"evoxc_hf-{hf_version}.dist-info"}
    require(tops == expected, f"{hf_wheel}: top-level entries must be exactly {sorted(expected)}, "
                              f"got {sorted(tops)}")
    compiled = sorted(entry for entry in entries if entry.endswith((".so", ".pyd", ".dylib")))
    require(not compiled, f"{hf_wheel}: ships compiled artifacts ({compiled[:5]}) — `evoxc-hf` is pure Python")

    # 5. The `evoxc` sdist has a single `evoxc-<version>` root, carries the Rust workspace needed to
    #    build the extension, and no `evoxc_hf` path at all.
    artifact = directory / f"evoxc-{version}.tar.gz"
    entries = sdist_entries(artifact)
    root = f"evoxc-{version}"
    require(top_level(entries) == {root}, f"{artifact}: must have the single root {root!r}, "
                                          f"got {sorted(top_level(entries))}")
    bad = sorted(entry for entry in entries if "evoxc_hf" in entry)
    require(not bad, f"{artifact}: ships {len(bad)} `evoxc_hf` path(s), e.g. {bad[:5]} — the workspace member "
                     "is a separate distribution and never part of the root sdist")
    for required in (f"{root}/native/Cargo.toml", f"{root}/native/evoxc-python/Cargo.toml"):
        require(required in entries, f"{artifact}: does not ship {required} — a source install must be able "
                                     "to build the extension from the sdist")

    # 6. The `evoxc_hf` sdist has a single `evoxc_hf-<version>` root, no `evoxc` package, and a license file.
    artifact = directory / f"evoxc_hf-{hf_version}.tar.gz"
    entries = sdist_entries(artifact)
    root = f"evoxc_hf-{hf_version}"
    require(top_level(entries) == {root}, f"{artifact}: must have the single root {root!r}, "
                                          f"got {sorted(top_level(entries))}")
    bad = sorted(entry for entry in entries if entry == f"{root}/evoxc" or entry.startswith(f"{root}/evoxc/"))
    require(not bad, f"{artifact}: ships the `evoxc` package ({bad[:5]}) — build `evoxc` from the workspace "
                     "ROOT project instead")
    require(any(entry.split("/")[-1] == "LICENSE" for entry in entries),
            f'{artifact}: does not ship the LICENSE file (license-files = ["LICENSE"] in evoxc_hf/pyproject.toml)')

    print(f"ok: evoxc {version} wheel/sdist ship `evoxc` + the abi3 extension only; "
          f"evoxc_hf {hf_version} wheel/sdist ship `evoxc_hf` only")


def manifest(version, hf):
    prefix = f"{'evoxc_hf' if hf else 'evoxc'}-{version}"
    names = [p.name for p in DIST.iterdir()]
    sdist = f"{prefix}.tar.gz"
    if hf:
        valid = set(names) == {sdist, f"{prefix}-py3-none-any.whl"}
    else:
        wheels = [name for name in names if re.fullmatch(re.escape(prefix) + r"-cp311-abi3-[a-z0-9_.]+\.whl", name)]
        valid = len(wheels) == 7 and set(names) == {*wheels, sdist}
    require(valid, f"Unexpected distributions for {prefix}: {names}")
    hashes = {}
    for name in sorted(names):
        path = DIST / name
        require(path.is_file() and not path.is_symlink(), f"Not a regular distribution file: {path}")
        with path.open("rb") as f:
            hashes[name] = hashlib.file_digest(f, "sha256").hexdigest()
    return hashes


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        # Never allow a redirected GitHub request to forward the workflow token.
        return None


def request(url, method="GET", headers=None, data=None):
    req = Request(url, data=data, headers=headers or {}, method=method)
    with build_opener(NoRedirect()).open(req, timeout=60) as response:
        return json.load(response)


def github(path, method="GET", payload=None, asset=None):
    headers = {"Authorization": f"Bearer {ENV['GH_TOKEN']}",
               "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    base = f"/repos/{ENV['GITHUB_REPOSITORY']}/{path}"
    url = "https://api.github.com" + base
    if asset is not None:
        headers.update({"Content-Type": "application/octet-stream", "Content-Length": str(asset.stat().st_size)})
        with asset.open("rb") as stream:
            return request("https://uploads.github.com" + base, method, headers, stream)
    if payload is not None:
        headers["Content-Type"] = "application/json"
    return request(url, method, headers, None if payload is None else json.dumps(payload).encode())


def pages(path):
    for page in range(1, 1000):
        items = github(f"{path}?per_page=100&page={page}")
        yield from items
        if len(items) < 100:
            return
    raise ReleaseError(f"Pagination limit exceeded for {path}")


def verify_live_tag(tag):
    obj = github(f"git/ref/tags/{quote(tag, safe='')}")
    require(obj["ref"] == f"refs/tags/{tag}" and obj["object"]["sha"] == ENV["TAG_OID"],
            f"Tag {tag} no longer matches the verified tag object")


def compare_subset(actual, expected, location):
    different = sorted(name for name in actual if actual[name] != expected.get(name))
    require(not different, f"{location} has unexpected or non-matching files: {different}; refusing to overwrite")
    return sorted(expected.keys() - actual.keys())


def release_assets(release_id):
    assets = {}
    for asset in pages(f"releases/{release_id}/assets"):
        name, digest = asset["name"], asset.get("digest") or ""
        require(asset["state"] == "uploaded" and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
                and name not in assets, f"GitHub asset {name!r} has no unique verified SHA-256; inspect it manually")
        assets[name] = digest.removeprefix("sha256:")
    return assets


def verify_release(release, tag, marker=None, *, release_id=None, allow_published=False):
    if release_id is not None:
        require(release["id"] == release_id, "GitHub Release ID does not match the staged release")
    require(release["tag_name"] == tag, "GitHub Release tag does not match")
    if marker is None:
        require(not release["draft"], "Expected a published HF release")
    else:
        require(allow_published or release["draft"], "GitHub Release is already published; refusing to mutate it")
        require(not release.get("prerelease"), "Unexpected GitHub prerelease; refusing to mutate it")
        require(marker in (release["body"] or ""),
                "GitHub Release marker does not match the verified source and artifacts; refusing to mutate it")


def evoxc_marker(expected):
    manifest_hash = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    return f"<!-- evoxc-release {ENV['CHECKOUT_SHA']} {ENV['TAG_OID']} {manifest_hash} -->"


def checked_evoxc_release(release_id, tag, marker, expected, *, allow_published=False):
    release = github(f"releases/{release_id}")
    verify_release(release, tag, marker, release_id=release_id, allow_published=allow_published)
    missing = compare_subset(release_assets(release_id), expected, "GitHub Release")
    require(not missing, f"GitHub Release no longer contains all tested assets; missing {missing}")
    return release


def stage_assets(release_id, tag, expected, marker=None):
    missing = compare_subset(release_assets(release_id), expected, "GitHub Release")
    for name in missing:
        verify_release(github(f"releases/{release_id}"), tag, marker)
        query = urlencode({"name": name})
        github(f"releases/{release_id}/assets?{query}", "POST", asset=DIST / name)
    require(not compare_subset(release_assets(release_id), expected, "GitHub Release"),
            "GitHub Release asset upload is incomplete")
    verify_release(github(f"releases/{release_id}"), tag, marker)


def pypi_files(project, version):
    try:
        url = f"https://pypi.org/pypi/{project}/{quote(version, safe='')}/json?check={time.time_ns()}"
        release = request(url, headers={"Cache-Control": "no-cache"})
    except HTTPError as exc:
        if exc.code != 404:
            raise
        exc.close()
        return None
    files = {}
    for item in release["urls"]:
        name = item["filename"]
        require(not item["yanked"] and name not in files, f"PyPI file {name!r} is yanked or duplicated")
        files[name] = item["digests"]["sha256"]
    return files


def preflight_pypi(project, version, expected):
    actual = pypi_files(project, version)
    if actual is not None:
        missing = compare_subset(actual, expected, "PyPI")
        require(not missing, f"Partial PyPI release; missing {missing}. Recover using only the missing files "
                "from the original tested artifact, then rerun failed jobs. No upload attempted.")
        print("PyPI already has exactly the tested distributions; skipping upload")
    output("upload", "true" if actual is None else "false")


def wait_for_pypi(project, version, expected):
    # The public index can lag a successful uploader; only retry missing files.
    for attempt in range(7):
        missing = compare_subset(pypi_files(project, version) or {}, expected, "PyPI")
        if not missing:
            return
        if attempt < 6:
            time.sleep(10)
    raise ReleaseError(f"PyPI does not yet contain all tested files ({missing}); check the index, then rerun only "
                       "the failed jobs in the original release run to reuse the tested artifact")


def main(command, dist_dir=None):
    if command == "dist":
        return dist_check(Path(dist_dir) if dist_dir else DIST)
    if command == "verify":
        return verify_source()
    version = ENV["VERSION"]
    hf = ENV["GITHUB_EVENT_NAME"] == "release"
    expected = manifest(version, hf)
    if command == "manifest":
        return
    tag = ENV["TAG"]
    verify_live_tag(tag)
    project = "evoxc-hf" if hf else "evoxc"
    if command == "preflight":
        release_id = int(ENV["RELEASE_ID"])
        if hf:
            verify_release(github(f"releases/{release_id}"), tag)
            compare_subset(release_assets(release_id), expected, "GitHub Release")
        else:
            checked_evoxc_release(release_id, tag, evoxc_marker(expected), expected)
            verify_live_tag(tag)
        return preflight_pypi(project, version, expected)
    if command == "attach-hf":
        wait_for_pypi(project, version, expected)
        verify_live_tag(tag)
        return stage_assets(int(ENV["RELEASE_ID"]), tag, expected)

    marker = evoxc_marker(expected)
    if command == "prepare":
        matches = [release for release in pages("releases") if release["tag_name"] == tag]
        require(len(matches) <= 1, f"Multiple releases claim {tag}; inspect them manually")
        release = matches[0] if matches else github(
            "releases", "POST", {"tag_name": tag, "target_commitish": ENV["CHECKOUT_SHA"], "draft": True,
                                 "name": tag, "body": marker, "generate_release_notes": True})
        verify_release(release, tag, marker)
        stage_assets(release["id"], tag, expected, marker)
        print(f"Verified draft: {release['html_url']}")
        output("release_id", release["id"])
    elif command == "finalize":
        wait_for_pypi(project, version, expected)
        release_id = int(ENV["RELEASE_ID"])
        release = checked_evoxc_release(release_id, tag, marker, expected, allow_published=True)
        verify_live_tag(tag)
        if not release["draft"]:
            print(f"Already published with the tested distributions: {release['html_url']}")
            return
        release = github(f"releases/{release_id}", "PATCH", {"draft": False})
        verify_release(release, tag, marker, release_id=release_id, allow_published=True)
        require(not release["draft"], "GitHub did not publish the expected release")
        print(f"Published {release['html_url']}")
    else:
        raise ReleaseError(f"Unknown command: {command}")


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    except (ReleaseError, HTTPError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
