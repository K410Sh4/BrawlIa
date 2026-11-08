"""Regression tests for dataset queue lifecycle and accounting."""

import tempfile
import unittest
from unittest.mock import patch

from brawlla.recognition_v1.dataset_store import DatasetStore


class DatasetIntegrityTests(unittest.TestCase):
    def test_submit_after_stop_does_not_enqueue(self):
        with tempfile.TemporaryDirectory() as folder:
            store = DatasetStore(folder)
            store._thread = object()  # No worker needed for submit lifecycle
            store.stop()
            with self.assertRaises(RuntimeError):
                store.submit({"frame_id": 1}, [])
            self.assertEqual(store.queue.qsize(), 0)
            self.assertEqual(store.snapshot()["accepted"], 0)

    def test_accepted_and_pending_are_reconciled(self):
        with tempfile.TemporaryDirectory() as folder:
            store = DatasetStore(folder)
            store._thread = object()
            with patch.object(
                store.curator, "select_samples", return_value=(False, "none")
            ):
                store.submit({"frame_id": 1}, [])
                store.submit({"frame_id": 2}, [])
            snapshot = store.snapshot()
            self.assertEqual(snapshot["accepted"], 2)
            self.assertEqual(snapshot["pending"], 2)
            self.assertEqual(snapshot["queue_depth"], 2)
            store.records = 1
            self.assertEqual(store.snapshot()["pending"], 1)

    def test_stop_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            store = DatasetStore(folder)
            store.stop()
            store.stop()
            self.assertTrue(store._stop.is_set())


if __name__ == "__main__":
    unittest.main()
