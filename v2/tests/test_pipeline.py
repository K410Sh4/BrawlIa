"""No game process, GPU or Windows required for deterministic core checks."""

import time
import unittest

import numpy as np

from brawlia_v2.capture import SyntheticSource
from brawlia_v2.pipeline import Frame, LatestFrame, Metrics, Pipeline, verify_frame


class LatestFrameTests(unittest.TestCase):
    def test_fast_producer_replaces_unconsumed_frame(self):
        slot = LatestFrame()
        for i in range(6):
            slot.put(Frame(i, time.perf_counter_ns(), None))
        result = slot.take(timeout=0)
        self.assertEqual(result.number, 5)
        self.assertEqual(slot.replacements, 5)
        self.assertIsNone(slot.take(timeout=0))
        slot.close()

    def test_close_wakes_consumer_without_fake_frame(self):
        slot = LatestFrame()
        slot.close()
        self.assertIsNone(slot.take(timeout=0))


class CapturePipelineTests(unittest.TestCase):
    def test_synthetic_30fps_producer_and_consumer(self):
        pipeline = Pipeline(SyntheticSource).start()
        time.sleep(1.2)
        metrics = pipeline.stop()
        self.assertIsNone(metrics["error"])
        self.assertGreater(metrics["capture_count"], 20)
        self.assertGreater(metrics["process_count"], 20)
        self.assertLess(metrics["capture_recent_fps"], 38)
        self.assertGreater(metrics["capture_recent_fps"], 22)
        self.assertFalse(metrics["png_recorded"])
        self.assertFalse(metrics["video_recorded"])
        self.assertGreater(metrics["visual_samples_checked"], 20)
        self.assertGreater(metrics["visually_unchanged_samples"], 0)

    def test_slow_processor_drops_old_frames_without_queue_growth(self):
        def slow(frame):
            verify_frame(frame)
            time.sleep(0.08)

        pipeline = Pipeline(SyntheticSource, process=slow).start()
        time.sleep(0.7)
        metrics = pipeline.stop()
        self.assertGreater(metrics["replaced_before_processing"], 0)
        self.assertLess(metrics["process_count"], metrics["capture_count"])
        self.assertLess(metrics["frame_age_p95_ms"], 350)

    def test_invalid_frame_reports_error_without_hanging(self):
        class BadSource:
            def open(self):
                pass

            def grab(self):
                return np.zeros((10, 20, 3), dtype=np.uint8)

            def close(self):
                pass

        pipeline = Pipeline(BadSource).start()
        time.sleep(0.15)
        metrics = pipeline.stop()
        self.assertIn("Quadro BGRA inválido", metrics["error"])

    def test_snapshot_is_bounded_and_p95_is_accurate(self):
        metric = Metrics()
        now = time.perf_counter_ns()
        for i in range(200):
            metric.captured(now + i * 33_333_333, i % 20 + 1, 0)
        snap = metric.snapshot()
        self.assertEqual(snap["capture_count"], 200)
        self.assertLessEqual(snap["capture_p95_ms"], 20)


if __name__ == "__main__":
    unittest.main()
