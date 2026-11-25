"""Local tests: python3 -m unittest discover -s .github/workflows -p 'test_release_checks.py'."""

import io
import json
import os
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import call, patch
from urllib.error import HTTPError, URLError

import release_checks as release

SHA = "a" * 40
TAG_OID = "b" * 40
TAG = "v1.2.3"
REF = f"refs/tags/{TAG}"
EXPECTED = {"wheel.whl": "a" * 64, "source.tar.gz": "b" * 64}


class SourceTests(unittest.TestCase):
    def verify(self, *, kind="push", ref=REF, sha=SHA, tag_oid=TAG_OID,
               event=None, ancestor_status=0):
        event = {"created": True, "deleted": False, "ref": REF, "after": TAG_OID} if event is None else event
        env = {"GITHUB_EVENT_PATH": "event.json", "GITHUB_EVENT_NAME": kind,
               "GITHUB_REF": ref, "GITHUB_SHA": sha}
        refs = {"HEAD^{commit}": SHA, REF: tag_oid, REF + "^{commit}": SHA}
        texts = {"event.json": json.dumps(event), "pyproject.toml": '[project]\nversion = "1.2.3"'}
        if kind == "release":
            refs.update({ref: tag_oid, ref + "^{commit}": SHA})
            texts["evoxc_hf/pyproject.toml"] = '[project]\nversion = "4.5.6"'
        with (patch.object(release, "ENV", env),
              patch.object(Path, "read_text", lambda path: texts[str(path)]),
              patch.object(release, "oid", side_effect=refs.__getitem__),
              patch.object(release.subprocess, "run") as ancestor,
              patch.object(release, "output") as output):
            ancestor.return_value.returncode = ancestor_status
            release.verify_source()
        return dict(call.args for call in output.call_args_list)

    def test_annotated_tag(self):
        self.assertEqual(self.verify(), {"checkout_sha": SHA, "tag": TAG,
                                        "tag_oid": TAG_OID, "version": "1.2.3"})
        self.assertEqual(self.verify(sha=TAG_OID)["checkout_sha"], SHA)

    def test_lightweight_tag(self):
        event = {"created": True, "deleted": False, "ref": REF, "after": SHA}
        self.assertEqual(self.verify(tag_oid=SHA, event=event)["tag_oid"], SHA)

    def test_rejects_moved_non_main_wrong_version_and_updated_tags(self):
        with self.assertRaisesRegex(release.ReleaseError, "does not match the event"):
            self.verify(tag_oid="c" * 40)
        with self.assertRaisesRegex(release.ReleaseError, "origin/main"):
            self.verify(ancestor_status=1)
        with self.assertRaisesRegex(release.ReleaseError, "Expected exact tag ref"):
            ref = "refs/tags/v1.2.4"
            self.verify(ref=ref, event={"created": True, "deleted": False, "ref": ref, "after": SHA})
        with self.assertRaisesRegex(release.ReleaseError, "Not a new tag push"):
            self.verify(event={"created": False, "deleted": False, "ref": REF, "after": SHA})

    def test_manual_uses_only_main_event_commit(self):
        self.assertEqual(self.verify(kind="workflow_dispatch", ref="refs/heads/main")["tag"], "")
        with self.assertRaisesRegex(release.ReleaseError, "main only"):
            self.verify(kind="workflow_dispatch", ref=REF)
        with self.assertRaisesRegex(release.ReleaseError, "does not match the event"):
            self.verify(kind="workflow_dispatch", ref="refs/heads/main", sha=TAG_OID)

    def test_hf_uses_member_version_without_main_gate(self):
        tag = "evoxc-hf-v4.5.6"
        event = {"action": "published", "release": {"tag_name": tag, "draft": False}}
        self.assertEqual(self.verify(kind="release", ref=f"refs/tags/{tag}", event=event,
                                     ancestor_status=1)["version"], "4.5.6")


class ArtifactTests(unittest.TestCase):
    def test_manifest_rejects_empty_extra_and_nonregular_files(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(release, "DIST", Path(tmp)):
            with self.assertRaisesRegex(release.ReleaseError, "Unexpected distributions"):
                release.manifest("1.2.3", hf=True)
            wheel = Path(tmp, "evoxc_hf-1.2.3-py3-none-any.whl")
            sdist = Path(tmp, "evoxc_hf-1.2.3.tar.gz")
            wheel.write_bytes(b"wheel")
            sdist.write_bytes(b"sdist")
            self.assertEqual(set(release.manifest("1.2.3", hf=True)), {wheel.name, sdist.name})
            extra = Path(tmp, "extra")
            extra.touch()
            with self.assertRaisesRegex(release.ReleaseError, "Unexpected distributions"):
                release.manifest("1.2.3", hf=True)
            extra.unlink()
            wheel.unlink()
            wheel.symlink_to(sdist)
            with self.assertRaisesRegex(release.ReleaseError, "Not a regular distribution file"):
                release.manifest("1.2.3", hf=True)

    def test_manifest_requires_a_flat_dist_directory(self):
        version = "1.2.3"
        platforms = ("manylinux2014_x86_64", "musllinux_1_2_x86_64",
                     "manylinux2014_aarch64", "musllinux_1_2_aarch64",
                     "macosx_10_12_x86_64", "macosx_11_0_arm64", "win_amd64")
        wheels = [f"evoxc-{version}-cp311-abi3-{platform}.whl" for platform in platforms]
        sdist = f"evoxc-{version}.tar.gz"
        with tempfile.TemporaryDirectory() as tmp, patch.object(release, "DIST", Path(tmp)):
            for name in (*wheels, sdist):
                Path(tmp, name).write_bytes(b"artifact")
            self.assertEqual(set(release.manifest(version, hf=False)), {*wheels, sdist})
            # A flat-but-wrong set keeps the generic message.
            Path(tmp, "extra").touch()
            with self.assertRaisesRegex(release.ReleaseError, "Unexpected distributions"):
                release.manifest(version, hf=False)
            Path(tmp, "extra").unlink()
            # actions/download-artifact@v4 nests each artifact unless merge-multiple is set.
            nested = Path(tmp, "evoxc-dist")
            nested.mkdir()
            Path(nested, sdist).write_bytes(b"artifact")
            for hf in (False, True):
                with (self.subTest(hf=hf),
                      self.assertRaisesRegex(release.ReleaseError,
                                             r"FLAT; found nested directory/directories \['evoxc-dist'\]")):
                    release.manifest(version, hf=hf)
            with self.assertRaisesRegex(release.ReleaseError, "merge-multiple: true"):
                release.manifest(version, hf=False)


WORKFLOW_DIR = Path(__file__).resolve().parent
DOWNLOAD_ARTIFACT = "actions/download-artifact@"


def download_artifact_steps(text):
    """Return the `with:` inputs of every actions/download-artifact step (indentation-aware scan)."""
    lines = text.splitlines()
    steps = []
    for index, line in enumerate(lines):
        if not line.strip().startswith("- uses:") or DOWNLOAD_ARTIFACT not in line:
            continue
        step_indent = len(line) - len(line.lstrip())
        inputs, in_with = {}, False
        for follower in lines[index + 1:]:
            body = follower.strip()
            if not body or body.startswith("#"):
                continue
            indent = len(follower) - len(follower.lstrip())
            if indent <= step_indent:
                break
            if indent == step_indent + 2:
                in_with = body == "with:"
                continue
            if in_with:
                key, _, value = body.partition(":")
                inputs[key.strip()] = value.strip().strip("'\"")
        steps.append(inputs)
    return steps


def writes_into_dist(inputs):
    path = inputs.get("path", "").removeprefix("./").rstrip("/")
    return path.split("/")[0] == "dist"


class WorkflowWiringTests(unittest.TestCase):
    def test_download_artifact_steps_extract_flat_into_dist(self):
        # release_checks.py treats dist/ as a FLAT directory (manifest, stage_assets, Twine), while
        # actions/download-artifact@v4 extracts each artifact into dist/<artifact-name>/ unless
        # merge-multiple: true is set. So a step writing into dist/ must either download ONE named
        # artifact (`name:`) or merge its artifacts (`merge-multiple: true`).
        workflows = sorted(WORKFLOW_DIR.glob("*.yml"))
        self.assertTrue(workflows)
        checked = []
        for path in workflows:
            for inputs in download_artifact_steps(path.read_text()):
                if not writes_into_dist(inputs):
                    continue
                checked.append(path.name)
                with self.subTest(workflow=path.name, inputs=inputs):
                    self.assertTrue("name" in inputs or inputs.get("merge-multiple") == "true",
                                    f"{path.name}: a download-artifact step writing into dist/ must set "
                                    f"`name:` or `merge-multiple: true`, got {inputs}")
        self.assertGreaterEqual(len(checked), 7)
        self.assertIn("release.yml", checked)
        self.assertIn("wheels.yml", checked)


class RemoteTests(unittest.TestCase):
    def test_pypi_upload_requires_absent_release_and_complete_match_skips(self):
        for actual, upload in [(None, "true"), (EXPECTED, "false")]:
            with (self.subTest(actual=actual), patch.object(release, "pypi_files", return_value=actual),
                  patch.object(release, "output") as output):
                release.preflight_pypi("evoxc", "1.2.3", EXPECTED)
                output.assert_called_once_with("upload", upload)

    def test_partial_empty_mismatched_and_extra_pypi_files_fail_closed(self):
        for actual in [dict(list(EXPECTED.items())[:1]), {}, {"wheel.whl": "c" * 64},
                       {**EXPECTED, "extra.whl": "d" * 64}]:
            with (self.subTest(actual=actual), patch.object(release, "pypi_files", return_value=actual),
                  patch.object(release, "output") as output):
                with self.assertRaises(release.ReleaseError):
                    release.preflight_pypi("evoxc", "1.2.3", EXPECTED)
                output.assert_not_called()

    def test_pypi_non_404_http_errors_are_not_treated_as_absent(self):
        for status in [403, 429, 503]:
            with (self.subTest(status=status), HTTPError("", status, "", {}, None) as error,
                  patch.object(release, "request", side_effect=error)):
                with self.assertRaises(HTTPError):
                    release.pypi_files("evoxc", "1.2.3")

    def test_matching_draft_uploads_only_missing_assets(self):
        marker = "<!-- managed -->"
        draft = {"tag_name": TAG, "draft": True, "body": marker}
        with (patch.object(release, "release_assets", side_effect=[{"wheel.whl": EXPECTED["wheel.whl"]}, EXPECTED]),
              patch.object(release, "github", return_value=draft) as github):
            release.stage_assets(17, TAG, EXPECTED, marker)
            posts = [call for call in github.call_args_list if "POST" in call.args]
            self.assertEqual(len(posts), 1)
            self.assertEqual(posts[0].args[0], "releases/17/assets?name=source.tar.gz")

    def test_does_not_mutate_published_or_nonmatching_draft(self):
        for draft, message in [(False, "already published"), (True, "marker does not match")]:
            remote = {"tag_name": TAG, "draft": draft, "body": "other"}
            with (self.subTest(draft=draft), patch.object(release, "release_assets", return_value={}),
                  patch.object(release, "github", return_value=remote) as github):
                with self.assertRaisesRegex(release.ReleaseError, message):
                    release.stage_assets(17, TAG, EXPECTED, "marker")
                github.assert_called_once_with("releases/17")


class EvoxcPublicationTests(unittest.TestCase):
    def setUp(self):
        env = {"VERSION": "1.2.3", "GITHUB_EVENT_NAME": "push", "TAG": TAG,
               "TAG_OID": TAG_OID, "CHECKOUT_SHA": SHA, "RELEASE_ID": "17"}
        self.enterContext(patch.object(release, "ENV", env))
        self.enterContext(patch.object(release, "manifest", return_value=EXPECTED))
        self.enterContext(patch.object(release.time, "sleep"))
        self.output = self.enterContext(patch.object(release, "output"))
        self.pypi_files = self.enterContext(patch.object(release, "pypi_files", return_value=EXPECTED))
        self.github = self.enterContext(patch.object(release, "github", side_effect=self.remote))
        self.staged = {"id": 17, "tag_name": TAG, "draft": True, "prerelease": False,
                       "body": release.evoxc_marker(EXPECTED) + "\n\nRelease notes",
                       "html_url": "https://github.com/example/project/releases/tag/v1.2.3"}
        self.assets = EXPECTED.copy()
        self.tag_oids = iter([TAG_OID] * 4)
        self.lose_response = False

    def remote(self, path, method="GET", payload=None):
        if (path, method) == ("git/ref/tags/" + TAG, "GET"):
            return {"ref": REF, "object": {"sha": next(self.tag_oids)}}
        if (path, method) == ("releases/17", "GET"):
            return self.staged.copy()
        if (path, method) == ("releases/17/assets?per_page=100&page=1", "GET"):
            return [{"name": name, "state": "uploaded", "digest": "sha256:" + digest}
                    for name, digest in self.assets.items()]
        if (path, method, payload) == ("releases/17", "PATCH", {"draft": False}):
            self.staged["draft"] = False
            if self.lose_response:
                raise URLError("lost PATCH response")
            return self.staged.copy()
        self.fail(f"Unexpected GitHub request: {method} {path}")

    def test_preflight_rechecks_the_pinned_draft_before_allowing_upload(self):
        self.pypi_files.return_value = None
        release.main("preflight")
        self.assertEqual(self.github.call_args_list, [call("git/ref/tags/" + TAG), call("releases/17"),
                         call("releases/17/assets?per_page=100&page=1"), call("git/ref/tags/" + TAG)])
        self.pypi_files.assert_called_once_with("evoxc", "1.2.3")
        self.output.assert_called_once_with("upload", "true")

    def test_preflight_rejects_public_release_before_upload(self):
        self.staged["draft"] = False
        with self.assertRaisesRegex(release.ReleaseError, "already published"):
            release.main("preflight")
        self.pypi_files.assert_not_called()
        self.output.assert_not_called()

    def test_preflight_and_recovery_reject_release_or_asset_drift(self):
        cases = [({"id": 18}, EXPECTED, "ID does not match"),
                 ({"tag_name": "v1.2.4"}, EXPECTED, "tag does not match"),
                 ({"body": "other"}, EXPECTED, "marker does not match"),
                 ({"prerelease": True}, EXPECTED, "Unexpected GitHub prerelease"),
                 ({}, {}, "missing"),
                 ({}, {**EXPECTED, "wheel.whl": "c" * 64}, "non-matching files"),
                 ({}, {**EXPECTED, "other.whl": "c" * 64}, "unexpected or non-matching files")]
        original = self.staged.copy()
        for command in ("preflight", "finalize"):
            for changes, assets, message in cases:
                with self.subTest(command=command, changes=changes, assets=assets):
                    self.tag_oids = iter([TAG_OID])
                    self.staged = {**original, "draft": command == "preflight", **changes}
                    self.assets = assets
                    self.github.reset_mock()
                    self.pypi_files.reset_mock()
                    with self.assertRaisesRegex(release.ReleaseError, message):
                        release.main(command)
                    self.assertTrue(all(len(c.args) == 1 for c in self.github.call_args_list))
                    if command == "preflight":
                        self.pypi_files.assert_not_called()
                    self.output.assert_not_called()

    def test_finalize_publishes_the_verified_draft(self):
        release.main("finalize")
        self.github.assert_called_with("releases/17", "PATCH", {"draft": False})
        self.assertFalse(self.staged["draft"])
        self.output.assert_not_called()

    def test_finalize_can_recover_a_lost_publish_response_without_mutating(self):
        self.lose_response = True
        with self.assertRaisesRegex(URLError, "lost PATCH response"):
            release.main("finalize")
        self.assertEqual(self.github.call_args_list[-2:], [call("git/ref/tags/" + TAG),
                         call("releases/17", "PATCH", {"draft": False})])
        self.github.reset_mock()
        self.pypi_files.reset_mock()
        release.main("finalize")
        self.assertEqual(self.github.call_args_list, [call("git/ref/tags/" + TAG), call("releases/17"),
                         call("releases/17/assets?per_page=100&page=1"), call("git/ref/tags/" + TAG)])
        self.pypi_files.assert_called_once_with("evoxc", "1.2.3")
        self.output.assert_not_called()

    def test_finalize_rejects_tag_moved_during_remote_checks(self):
        for draft in (True, False):
            with self.subTest(draft=draft):
                self.staged["draft"] = draft
                self.tag_oids = iter([TAG_OID, "c" * 40])
                self.github.reset_mock()
                with self.assertRaisesRegex(release.ReleaseError, "no longer matches the verified tag object"):
                    release.main("finalize")
                self.assertEqual(self.github.call_count, 4)
                self.github.assert_called_with("git/ref/tags/" + TAG)
                self.assertIs(self.staged["draft"], draft)

    def test_already_public_recovery_still_requires_exact_pypi(self):
        self.staged["draft"] = False
        for files, message in [(None, "all tested files"), ({}, "all tested files"),
                               ({**EXPECTED, "wheel.whl": "c" * 64}, "non-matching files"),
                               ({**EXPECTED, "extra.whl": "c" * 64}, "unexpected or non-matching")]:
            with self.subTest(files=files):
                self.tag_oids = iter([TAG_OID])
                self.pypi_files.return_value = files
                self.github.reset_mock()
                with self.assertRaisesRegex(release.ReleaseError, message):
                    release.main("finalize")
                self.github.assert_called_once_with("git/ref/tags/" + TAG)
                self.output.assert_not_called()


class DistTests(unittest.TestCase):
    VERSION = "1.2.3"
    HF_VERSION = "4.5.6"
    WHEEL = f"evoxc-{VERSION}-cp311-abi3-linux_x86_64.whl"
    WHEEL_ENTRIES = ["evoxc/__init__.py", "evoxc/_evoxc_native.abi3.so",
                     f"evoxc-{VERSION}.dist-info/METADATA"]
    SDIST = f"evoxc-{VERSION}.tar.gz"
    SDIST_ENTRIES = [f"evoxc-{VERSION}/pyproject.toml", f"evoxc-{VERSION}/native/Cargo.toml",
                     f"evoxc-{VERSION}/native/evoxc-python/Cargo.toml"]
    HF_WHEEL = f"evoxc_hf-{HF_VERSION}-py3-none-any.whl"
    HF_WHEEL_ENTRIES = ["evoxc_hf/__init__.py", f"evoxc_hf-{HF_VERSION}.dist-info/METADATA"]
    HF_SDIST = f"evoxc_hf-{HF_VERSION}.tar.gz"
    HF_SDIST_ENTRIES = [f"evoxc_hf-{HF_VERSION}/LICENSE", f"evoxc_hf-{HF_VERSION}/pyproject.toml"]

    def setUp(self):
        self.old_cwd = os.getcwd()
        self.addCleanup(os.chdir, self.old_cwd)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def wheel(self, directory, name, entries):
        with zipfile.ZipFile(directory / name, "w") as archive:
            for entry in entries:
                archive.writestr(entry, "")

    def sdist(self, directory, name, entries):
        with tarfile.open(directory / name, "w:gz") as archive:
            for entry in entries:
                info = tarfile.TarInfo(entry)
                info.size = 0
                archive.addfile(info, io.BytesIO(b""))

    def build(self, directory):
        self.wheel(directory, self.WHEEL, self.WHEEL_ENTRIES)
        self.sdist(directory, self.SDIST, self.SDIST_ENTRIES)
        self.wheel(directory, self.HF_WHEEL, self.HF_WHEEL_ENTRIES)
        self.sdist(directory, self.HF_SDIST, self.HF_SDIST_ENTRIES)

    def fixture(self):
        """Build a fresh valid four-archive tree in a new directory and chdir into it."""
        root = Path(tempfile.mkdtemp(dir=self.tmp.name))
        os.chdir(root)
        Path(root, "pyproject.toml").write_text(f'[project]\nversion = "{self.VERSION}"\n')
        Path(root, "evoxc_hf").mkdir()
        Path(root, "evoxc_hf", "pyproject.toml").write_text(f'[project]\nversion = "{self.HF_VERSION}"\n')
        directory = Path(root, "dist")
        directory.mkdir()
        self.build(directory)
        return directory

    def assert_dist_fails(self, mutation, message):
        with self.subTest(mutation=mutation.__name__):
            directory = self.fixture()
            mutation(directory)
            with self.assertRaisesRegex(release.ReleaseError, message):
                release.dist_check(directory)

    def test_valid_tree_passes_with_default_and_explicit_directory(self):
        directory = self.fixture()
        release.dist_check()
        release.dist_check(directory)
        release.dist_check(str(directory))
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            release.dist_check(directory)
        self.assertEqual(stdout.getvalue().splitlines(), [
            f"ok: evoxc {self.VERSION} wheel/sdist ship `evoxc` + the abi3 extension only; "
            f"evoxc_hf {self.HF_VERSION} wheel/sdist ship `evoxc_hf` only"])

    def test_main_dispatches_dist_without_release_environment(self):
        directory = self.fixture()
        with patch.object(release, "ENV", {}):
            release.main("dist", str(directory))
            release.main("dist")

    def test_accepts_maturin_bundled_shared_libraries(self):
        variants = (
            (self.WHEEL, []),
            (f"evoxc-{self.VERSION}-cp311-abi3-musllinux_1_2_x86_64.whl",
             ["evoxc.libs/libgcc_s-f685abf1.so.1"]),
            (f"evoxc-{self.VERSION}-cp311-abi3-musllinux_1_2_aarch64.whl",
             ["evoxc.libs/libgcc_s-0bf60adc.so.1"]),
            (f"evoxc-{self.VERSION}-cp311-abi3-manylinux_2_17_x86_64.whl",
             ["evoxc.libs/", "evoxc.libs/libfoo.so", "evoxc.libs/libbar.so.1",
              "evoxc.libs/libbaz.dylib", "evoxc.libs/libqux.dll"]),
        )
        for name, bundled in variants:
            with self.subTest(wheel=name, bundled=bundled):
                directory = self.fixture()
                (directory / self.WHEEL).unlink()
                self.wheel(directory, name, [*self.WHEEL_ENTRIES, *bundled])
                release.dist_check(directory)

    def test_rejects_non_shared_library_files_in_bundled_libs(self):
        def with_python_file(directory):
            self.wheel(directory, self.WHEEL, [*self.WHEEL_ENTRIES, "evoxc.libs/notes.py"])

        def with_nested_library(directory):
            self.wheel(directory, self.WHEEL, [*self.WHEEL_ENTRIES, "evoxc.libs/nested/libfoo.so"])

        def with_libs_file(directory):
            self.wheel(directory, self.WHEEL, [*self.WHEEL_ENTRIES, "evoxc.libs"])

        for mutation in (with_python_file, with_nested_library, with_libs_file):
            self.assert_dist_fails(mutation, "shared-library files")
        for entry in ("evoxc.libs/notes.py", "evoxc.libs/nested/libfoo.so", "evoxc.libs"):
            directory = self.fixture()
            self.wheel(directory, self.WHEEL, [*self.WHEEL_ENTRIES, entry])
            with self.assertRaises(release.ReleaseError) as caught:
                release.dist_check(directory)
            self.assertIn(entry, str(caught.exception))

    def test_rejects_bundled_library_directory_in_the_evoxc_hf_wheel(self):
        def with_bundled_library(directory):
            self.wheel(directory, self.HF_WHEEL, [*self.HF_WHEEL_ENTRIES, "evoxc_hf.libs/libfoo.so"])

        self.assert_dist_fails(with_bundled_library, "top-level entries must be exactly")

    def test_accepts_compressed_multi_tag_wheel_names(self):
        directory = self.fixture()
        (directory / self.WHEEL).unlink()
        self.wheel(directory,
                   f"evoxc-{self.VERSION}-cp311-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
                   self.WHEEL_ENTRIES)
        release.dist_check(directory)

    def test_rejects_wrong_evoxc_wheel_set(self):
        def duplicate_wheel(directory):
            self.wheel(directory, f"evoxc-{self.VERSION}-cp312-abi3-macosx_11_0_arm64.whl",
                       [f"evoxc-{self.VERSION}.dist-info/METADATA"])

        def drop_wheel(directory):
            (directory / self.WHEEL).unlink()

        def pure_python_wheel(directory):
            (directory / self.WHEEL).unlink()
            self.wheel(directory, f"evoxc-{self.VERSION}-py3-none-any.whl", self.WHEEL_ENTRIES)

        self.assert_dist_fails(duplicate_wheel, "exactly one platform")
        self.assert_dist_fails(drop_wheel, "exactly one platform")
        self.assert_dist_fails(pure_python_wheel, "platform-specific")

    def test_rejects_extra_or_missing_archives(self):
        def extra_wheel(directory):
            self.wheel(directory, "stray-1.0-py3-none-any.whl", ["stray/__init__.py"])

        def extra_sdist(directory):
            self.sdist(directory, "stray-1.0.tar.gz", ["stray-1.0/pyproject.toml"])

        def missing_hf_wheel(directory):
            (directory / self.HF_WHEEL).unlink()

        for mutation in (extra_wheel, extra_sdist):
            self.assert_dist_fails(mutation, "expected exactly 4")
        self.assert_dist_fails(missing_hf_wheel, "missing artifact")

    def test_rejects_evoxc_wheel_boundary_breaks(self):
        def with_hf(directory):
            self.wheel(directory, self.WHEEL, [*self.WHEEL_ENTRIES, "evoxc_hf/__init__.py"])

        def without_extension(directory):
            self.wheel(directory, self.WHEEL, [entry for entry in self.WHEEL_ENTRIES
                                               if not entry.endswith(".so")])

        def with_extra_top_level(directory):
            self.wheel(directory, self.WHEEL, [*self.WHEEL_ENTRIES, "other/__init__.py"])

        self.assert_dist_fails(with_hf, "evoxc_hf")
        self.assert_dist_fails(without_extension, "ships no evoxc/_evoxc_native.abi3.so")
        self.assert_dist_fails(with_extra_top_level, "top-level entries must be exactly")

    def test_rejects_evoxc_sdist_boundary_breaks(self):
        def wrong_root(directory):
            self.sdist(directory, self.SDIST, [f"evoxc-9.9.9/{entry.split('/', 1)[1]}"
                                               for entry in self.SDIST_ENTRIES])

        def with_hf_path(directory):
            self.sdist(directory, self.SDIST, [*self.SDIST_ENTRIES, f"evoxc-{self.VERSION}/evoxc_hf/pyproject.toml"])

        def without_native_cargo(directory):
            self.sdist(directory, self.SDIST, [entry for entry in self.SDIST_ENTRIES
                                               if entry != f"evoxc-{self.VERSION}/native/Cargo.toml"])

        self.assert_dist_fails(wrong_root, "single root")
        self.assert_dist_fails(with_hf_path, "evoxc_hf")
        self.assert_dist_fails(without_native_cargo, "native/Cargo.toml")

    def test_rejects_evoxc_hf_wheel_boundary_breaks(self):
        def with_evoxc(directory):
            self.wheel(directory, self.HF_WHEEL, [*self.HF_WHEEL_ENTRIES, "evoxc/__init__.py"])

        def with_compiled(directory):
            self.wheel(directory, self.HF_WHEEL, [*self.HF_WHEEL_ENTRIES, "evoxc_hf/_extension.so"])

        def with_extra_top_level(directory):
            self.wheel(directory, self.HF_WHEEL, [*self.HF_WHEEL_ENTRIES, "other/__init__.py"])

        self.assert_dist_fails(with_evoxc, "ships the `evoxc` package")
        self.assert_dist_fails(with_compiled, "compiled artifacts")
        self.assert_dist_fails(with_extra_top_level, "top-level entries must be exactly")

    def test_rejects_evoxc_hf_sdist_boundary_breaks(self):
        def without_license(directory):
            self.sdist(directory, self.HF_SDIST, [entry for entry in self.HF_SDIST_ENTRIES
                                                  if not entry.endswith("/LICENSE")])

        def with_evoxc(directory):
            self.sdist(directory, self.HF_SDIST, [*self.HF_SDIST_ENTRIES,
                                                  f"evoxc_hf-{self.HF_VERSION}/evoxc/__init__.py"])

        def wrong_root(directory):
            self.sdist(directory, self.HF_SDIST, [f"evoxc_hf-9.9.9/{entry.split('/', 1)[1]}"
                                                  for entry in self.HF_SDIST_ENTRIES])

        self.assert_dist_fails(without_license, "LICENSE")
        self.assert_dist_fails(with_evoxc, "ships the `evoxc` package")
        self.assert_dist_fails(wrong_root, "single root")


if __name__ == "__main__":
    unittest.main()
