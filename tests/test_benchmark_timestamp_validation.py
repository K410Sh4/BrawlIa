"""Regressões de validação temporal do benchmark de reconhecimento."""

import json
import tempfile
import unittest
from pathlib import Path

from qa.benchmark_recognition_v1 import benchmark


class BenchmarkTimestampTests(unittest.TestCase):
    def _run_with_timestamps(self, timestamps):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            records = [
                {"timestamp_s": value, "processing_ms": 1.0}
                for value in timestamps
            ]
            (folder / "observations.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            return benchmark(folder)

    def test_valid_timestamps(self):
        result = self._run_with_timestamps([1.0, 1.5, 2.0])
        self.assertEqual(result["cycles"], 3)
        self.assertEqual(result["effective_fps"], 2.0)

    def test_repeated_timestamp_rejected(self):
        with self.assertRaisesRegex(ValueError, "estritamente crescentes"):
            self._run_with_timestamps([1.0, 1.0])

    def test_backwards_timestamp_rejected(self):
        with self.assertRaisesRegex(ValueError, "estritamente crescentes"):
            self._run_with_timestamps([2.0, 1.0])

    def test_nonfinite_timestamp_rejected(self):
        with self.assertRaisesRegex(ValueError, "não finitos"):
            self._run_with_timestamps([1.0, float("nan")])


if __name__ == "__main__":
    unittest.main()
