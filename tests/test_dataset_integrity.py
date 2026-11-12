"""Testes offline de integridade de dataset, sem OpenCV/BlueStacks."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from brawlla.recognition_v1.dataset_integrity import reconcile_dataset_integrity


class DatasetIntegrityTests(unittest.TestCase):
    def test_valid_and_corrupted_image(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            image = root / "images" / "frame.png"
            image.write_bytes(b"png-test")
            entry = {
                "image": "images/frame.png",
                "sha256": hashlib.sha256(b"png-test").hexdigest(),
            }
            (root / "manifest.jsonl").write_text(json.dumps(entry) + "\n")
            self.assertTrue(reconcile_dataset_integrity(root)["ok"])
            image.write_bytes(b"changed")
            self.assertEqual(
                reconcile_dataset_integrity(root)["hash_mismatches"],
                ["images/frame.png"],
            )

    def test_missing_orphan_and_unsafe_paths(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "images" / "orphan.png").write_bytes(b"orphan")
            lines = [
                {"image": "images/missing.png", "sha256": "0" * 64},
                {"image": "../outside.png", "sha256": "0" * 64},
            ]
            (root / "manifest.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in lines)
            )
            report = reconcile_dataset_integrity(root)
            self.assertEqual(report["missing_images"], ["images/missing.png"])
            self.assertEqual(report["orphan_images"], ["images/orphan.png"])
            self.assertEqual(len(report["invalid_entries"]), 1)
            self.assertFalse(report["ok"])

    def test_missing_manifest(self):
        with TemporaryDirectory() as temporary:
            report = reconcile_dataset_integrity(Path(temporary))
            self.assertFalse(report["ok"])
            self.assertIn("manifest.jsonl ausente", report["invalid_entries"])


if __name__ == "__main__":
    unittest.main()
