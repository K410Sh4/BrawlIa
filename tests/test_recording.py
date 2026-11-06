"""Testes de gravação; usa clipes artificiais, não depende de BlueStacks."""

import json
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np

from brawlla.recording import AnalysisRecorder


class RecordingTests(unittest.TestCase):
    def test_records_video_and_detection_log(self):
        with tempfile.TemporaryDirectory() as temp:
            rec = AnalysisRecorder(Path(temp), fps=10)
            rec.start()
            for i in range(4):
                frame = np.full((240, 320, 3), (i * 30, 90, 150), dtype=np.uint8)
                cv2.putText(
                    frame,
                    "HP ALVO ?",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                rec.submit(
                    frame,
                    {
                        "red_health_bar_candidates": [{"x": i, "y": 2}],
                        "mode": "read_only",
                    },
                )
                time.sleep(0.13)
            rec.stop()
            self.assertTrue(rec.join(10), "Finalização não terminou")
            snap = rec.snapshot()
            self.assertIsNone(snap["error"])
            self.assertGreaterEqual(snap["written"], 3)
            self.assertGreaterEqual(snap["records"], 3)
            self.assertTrue(Path(snap["video"]).exists())
            reader = cv2.VideoCapture(snap["video"])
            try:
                self.assertTrue(reader.isOpened())
                self.assertEqual(int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), 320)
                self.assertEqual(int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)), 240)
                self.assertGreaterEqual(int(reader.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                self.assertTrue(reader.read()[0])
            finally:
                reader.release()
            entries = [
                json.loads(line)
                for line in rec.log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(entries), snap["records"])
            self.assertEqual(entries[0]["detections"]["mode"], "read_only")
            self.assertIn("timestamp_utc", entries[0])
            summary = json.loads(rec.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["video_frames"], snap["written"])

    def test_recorder_keeps_only_most_recent_frame_if_busy(self):
        with tempfile.TemporaryDirectory() as temp:
            rec = AnalysisRecorder(Path(temp), fps=5)
            img = np.zeros((240, 320, 3), np.uint8)
            rec.submit(img, {"index": 1})
            rec.submit(img, {"index": 2})
            self.assertEqual(rec.snapshot()["dropped"], 1)
            rec.start()
            rec.stop()
            self.assertTrue(rec.join(10))
            self.assertIsNone(rec.snapshot()["error"])
            records = [
                json.loads(line)
                for line in rec.log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["detections"]["index"], 2)

    def test_early_stop_even_without_any_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            rec = AnalysisRecorder(Path(temp), fps=15)
            rec.start()
            rec.stop()
            self.assertTrue(rec.join(10))
            self.assertEqual(rec.snapshot()["written"], 0)
            self.assertIsNone(rec.snapshot()["error"])
            self.assertTrue(rec.summary_path.exists())

    def test_odd_size_is_padded_for_video_codec(self):
        with tempfile.TemporaryDirectory() as temp:
            rec = AnalysisRecorder(Path(temp), fps=5)
            rec.start()
            rec.submit(np.zeros((241, 321, 3), np.uint8), {"test": "odd-resolution"})
            rec.stop()
            self.assertTrue(rec.join(10))
            self.assertIsNone(rec.snapshot()["error"])
            cap = cv2.VideoCapture(rec.snapshot()["video"])
            try:
                self.assertTrue(cap.isOpened())
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 322)
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 242)
            finally:
                cap.release()

    def test_rejects_invalid_fps(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                AnalysisRecorder(Path(temp), fps=-1)

    def test_disk_guard_records_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            rec = AnalysisRecorder(Path(temp), fps=5, min_free_bytes=10**18)
            rec.start()
            rec.submit(np.zeros((240, 320, 3), np.uint8), {"mode": "read_only"})
            rec.stop()
            self.assertTrue(rec.join(10))
            self.assertEqual(rec.snapshot()["written"], 0)
            self.assertIn("espaço", rec.snapshot()["limit_reached"])


if __name__ == "__main__":
    unittest.main()
