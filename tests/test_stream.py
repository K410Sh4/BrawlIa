import time
import unittest

import numpy as np

from brawlla.stream import LatestFrame, VisionPipeline, resize_for_analysis


class StreamTests(unittest.TestCase):
    def test_latest_frame_replaces_stale_frames(self):
        slot = LatestFrame()
        slot.publish(np.array([1]))
        slot.publish(np.array([2]))
        seq, image = slot.take_after(0, timeout=0)
        self.assertEqual(seq, 2)
        self.assertEqual(image[0], 2)
        slot.stop()

    def test_resize_limits_analysis_not_capture(self):
        raw = np.zeros((1080, 1920, 3), np.uint8)
        processed = resize_for_analysis(raw, 960)
        self.assertEqual(processed.shape, (540, 960, 3))
        self.assertEqual(raw.shape, (1080, 1920, 3))

    def test_pipeline_capture_independent_of_slow_analysis(self):
        result = []

        class FakeSource:
            def grab(self):
                return np.zeros((240, 320, 3), dtype=np.uint8)

            def close(self):
                pass

        def slow_analyze(frame):
            time.sleep(0.11)
            return frame, {}

        pipe = VisionPipeline(
            FakeSource,
            slow_analyze,
            lambda f, d: result.append(1),
            target_fps=30,
            analysis_fps=30,
            max_width=640,
        )
        pipe.start()
        time.sleep(0.8)
        pipe.stop()
        pipe.join(2)
        data = pipe.snapshot()
        self.assertGreater(data["captured"], data["analyzed"])
        self.assertGreater(data["skipped"], 0)
        self.assertEqual(data["error"], "")
        self.assertGreater(len(result), 0)

    def test_reject_invalid_fps(self):
        with self.assertRaises(ValueError):
            VisionPipeline(None, None, None, target_fps=999)

    def test_no_reprocessing_same_frame(self):
        slot = LatestFrame()
        slot.publish(np.zeros((10, 10, 3), np.uint8))
        seq, frame = slot.take_after(0, timeout=0)
        self.assertEqual(seq, 1)
        seq2, frame2 = slot.take_after(1, timeout=0.03)
        self.assertEqual(seq2, 1)
        self.assertIsNone(frame2)
        slot.stop()


if __name__ == "__main__":
    unittest.main()
