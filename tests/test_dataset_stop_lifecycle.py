"""Regressões para o ciclo de vida do DatasetStore (auditoria 2026-11-08)."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from brawlla.recognition_v1.dataset_store import DatasetStore


class DatasetStopTests(unittest.TestCase):
    def test_submit_after_stop_is_rejected_without_queue_growth(self):
        with TemporaryDirectory() as tmp:
            store = DatasetStore(Path(tmp) / "session", min_free_bytes=0)
            store.start()
            store.stop()
            before = store.queue.qsize()
            result = {
                "frame_id": 1,
                "timestamp_s": 0.0,
                "screen_resolution": [640, 360],
                "capture_mode": "regional",
                "tracks": [],
                "events": [],
            }
            with self.assertRaisesRegex(RuntimeError, "encerrando"):
                store.submit(result, [])
            self.assertEqual(store.queue.qsize(), before)
            store.join(5)
            self.assertFalse(store._thread.is_alive())

    def test_stop_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            store = DatasetStore(Path(tmp) / "session", min_free_bytes=0)
            store.start()
            store.stop()
            store.stop()
            store.join(5)
            self.assertFalse(store._thread.is_alive())
            self.assertIsNone(store.error)


if __name__ == "__main__":
    unittest.main()
