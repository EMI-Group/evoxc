"""Fail-closed release checks; run from the checkout (verify) or beside dist/ (other commands)."""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import tomllib
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


def verify_release(release, tag, marker=None):
    require(release["tag_name"] == tag, "GitHub Release tag does not match")
    if marker is None:
        require(not release["draft"], "Expected a published HF release")
    else:
        require(release["draft"] and marker in (release["body"] or ""),
                "Release already published or draft does not match the verified source and artifacts; refusing to mutate it")


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
    raise ReleaseError(f"PyPI does not yet contain all tested files ({missing}); rerun failed jobs after checking the index")


def main(command):
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
        if hf:
            release_id = int(ENV["RELEASE_ID"])
            verify_release(github(f"releases/{release_id}"), tag)
            compare_subset(release_assets(release_id), expected, "GitHub Release")
        return preflight_pypi(project, version, expected)
    if command == "attach-hf":
        wait_for_pypi(project, version, expected)
        verify_live_tag(tag)
        return stage_assets(int(ENV["RELEASE_ID"]), tag, expected)

    manifest_hash = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    sha = ENV["CHECKOUT_SHA"]
    marker = f"<!-- evoxc-release {sha} {ENV['TAG_OID']} {manifest_hash} -->"
    if command == "prepare":
        matches = [release for release in pages("releases") if release["tag_name"] == tag]
        require(len(matches) <= 1, f"Multiple releases claim {tag}; inspect them manually")
        release = matches[0] if matches else github(
            "releases", "POST", {"tag_name": tag, "target_commitish": sha, "draft": True,
                                 "name": tag, "body": marker, "generate_release_notes": True})
        verify_release(release, tag, marker)
        stage_assets(release["id"], tag, expected, marker)
        print(f"Verified draft: {release['html_url']}")
        output("release_id", release["id"])
    elif command == "finalize":
        wait_for_pypi(project, version, expected)
        release_id = int(ENV["RELEASE_ID"])
        path = f"releases/{release_id}"
        verify_release(github(path), tag, marker)
        require(not compare_subset(release_assets(release_id), expected, "GitHub draft"),
                "GitHub draft no longer contains the tested assets")
        verify_live_tag(tag)
        release = github(path, "PATCH", {"draft": False})
        require(not release["draft"] and release["tag_name"] == tag, "GitHub did not publish the expected release")
        print(f"Published {release['html_url']}")
    else:
        raise ReleaseError(f"Unknown command: {command}")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except (ReleaseError, HTTPError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
