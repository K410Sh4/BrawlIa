"""Regression tests for dataset summary write failures."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brawlla.recognition_v1.dataset_store import DatasetStore


class DatasetSummaryFailureTests(unittest.TestCase):
    def test_summary_write_failure_is_reported_without_thread_exception(self):
        with tempfile.TemporaryDirectory() as root:
            store = DatasetStore(Path(root) / "dataset")
            with patch.object(Path, "write_text", side_effect=OSError("disk full")):
                store.start()
                store.stop()
                store.join(timeout=5)
            self.assertFalse(store._thread.is_alive())
            self.assertIn("Falha ao salvar resumo", store.error)

    def test_original_worker_error_is_preserved_if_summary_also_fails(self):
        with tempfile.TemporaryDirectory() as root:
            store = DatasetStore(Path(root) / "dataset", max_bytes=0)
            with patch.object(Path, "write_text", side_effect=OSError("disk full")):
                store.start()
                store.submit(
                    {
                        "frame_id": 1,
                        "timestamp_s": 1.0,
                        "state": {"name": "combate"},
                        "events": [],
                    },
                    [],
                )
                store.stop()
                store.join(timeout=5)
            self.assertFalse(store._thread.is_alive())
            self.assertEqual(
                store.error, "Limite do dataset atingido pelas observacoes"
            )


if __name__ == "__main__":
    unittest.main()
