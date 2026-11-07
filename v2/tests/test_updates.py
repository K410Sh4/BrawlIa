"""OTA validation without network or running Windows executables."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brawlia_v2 import updates
from brawlia_v2.installer import apply_update


class ReleaseTests(unittest.TestCase):
    def _release(self, tag="v2.0.1"):
        base = f"https://github.com/{updates.REPO}/releases/download/{tag}"
        return {
            "tag_name": tag,
            "draft": False,
            "prerelease": False,
            "html_url": f"https://github.com/{updates.REPO}/releases/tag/{tag}",
            "assets": [
                {"name": updates.MANIFEST_NAME,
                 "browser_download_url": f"{base}/{updates.MANIFEST_NAME}"},
                {"name": updates.ASSET_NAME,
                 "browser_download_url": f"{base}/{updates.ASSET_NAME}"},
            ],
        }

    def test_new_release_selected(self):
        release = updates.parse_latest(self._release(), "2.0.0")
        self.assertEqual(release.version, "2.0.1")
        self.assertIn("github.com", release.executable_url)

    def test_old_and_foreign_releases_ignored(self):
        self.assertIsNone(updates.parse_latest(self._release("v2.0.0"), "2.0.0"))
        self.assertIsNone(updates.parse_latest(self._release("v1.9.0"), "2.0.0"))
        other = self._release()
        other["prerelease"] = True
        self.assertIsNone(updates.parse_latest(other, "2.0.0"))

    def test_manifest_digest_size_and_version_validation(self):
        release = updates.parse_latest(self._release(), "2.0.0")
        correct = {"version": "2.0.1", "filename": "BrawlIA-V2.exe",
                   "sha256": "a" * 64, "size": 16384}
        self.assertEqual(updates.validate_manifest(correct, release), ("a" * 64, 16384))
        for corrupted in (
            {**correct, "version": "2.0.0"},
            {**correct, "filename": "malware.exe"},
            {**correct, "size": 0},
            {**correct, "sha256": "not-a-hash"},
        ):
            with self.assertRaises(ValueError):
                updates.validate_manifest(corrupted, release)

    def test_reject_untrusted_url_and_plain_http(self):
        for url in ("http://github.com/x", "https://evil.example/x",
                    "file:///c:/temp/file", "https://user:pass@github.com/x"):
            with self.assertRaises(ValueError):
                updates.validate_https_url(url)

    def test_download_verified_and_corruption(self):
        release = updates.parse_latest(self._release(), "2.0.0")
        exe_data = b"EXE-test-content" * 100
        manifest = {"version": release.version, "filename": "BrawlIA-V2.exe",
                    "size": len(exe_data),
                    "sha256": hashlib.sha256(exe_data).hexdigest()}

        class Response:
            def __init__(self, body, url):
                self.data, self.url = body, url
                self.cursor = 0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return None
            def geturl(self):
                return self.url
            def read(self, size=-1):
                if size < 0:
                    size = len(self.data)
                chunk = self.data[self.cursor:self.cursor + size]
                self.cursor += len(chunk)
                return chunk

        def fake_open(req, timeout=None):
            url = req.full_url
            if url.endswith("ota-manifest.json"):
                return Response(json.dumps(manifest).encode(), url)
            return Response(exe_data, url)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(updates, "urlopen", side_effect=fake_open):
                staged, digest = updates.download_verified(release, tmp)
            self.assertEqual(staged.read_bytes(), exe_data)
            self.assertEqual(digest, manifest["sha256"])
            manifest["sha256"] = "0" * 64
            with patch.object(updates, "urlopen", side_effect=fake_open):
                with self.assertRaises(ValueError):
                    updates.download_verified(release, tmp)

    def test_sidecar_updater_install_with_backup_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            target = base / "BrawlIA-V2.exe"
            target.write_bytes(b"old")
            staged_dir = base / ".updates" / "2.0.1"
            staged_dir.mkdir(parents=True)
            staged = staged_dir / "BrawlIA-V2-2.0.1.exe"
            staged.write_bytes(b"new")
            sha = hashlib.sha256(b"new").hexdigest()
            result = apply_update(target, staged, sha, relaunch=False)
            self.assertEqual(result.read_bytes(), b"new")
            self.assertEqual((base / "BrawlIA-V2.previous.exe").read_bytes(), b"old")
            with self.assertRaises(ValueError):
                apply_update(target, staged, "0" * 64, relaunch=False)
            self.assertEqual(result.read_bytes(), b"new")


if __name__ == "__main__":
    unittest.main()
