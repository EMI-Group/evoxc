"""Local tests: python3 -m unittest discover -s .github/workflows -p 'test_release_checks.py'."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

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
        for draft in [{"tag_name": TAG, "draft": False, "body": "marker"},
                      {"tag_name": TAG, "draft": True, "body": "other"}]:
            with (self.subTest(draft=draft), patch.object(release, "release_assets", return_value={}),
                  patch.object(release, "github", return_value=draft) as github):
                with self.assertRaisesRegex(release.ReleaseError, "already published or draft does not match"):
                    release.stage_assets(17, TAG, EXPECTED, "marker")
                github.assert_called_once_with("releases/17")


if __name__ == "__main__":
    unittest.main()
