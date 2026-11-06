import tempfile
import time
from pathlib import Path
import unittest
import numpy as np
from brawlla.samples import SampleRecorder


class SampleTests(unittest.TestCase):
    def test_saves_unlabeled_png_with_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SampleRecorder(Path(tmp), sample_fps=15)
            recorder.start()
            for _ in range(4):
                recorder.submit(np.zeros((100, 160, 3), np.uint8))
                time.sleep(0.085)
            recorder.stop()
            recorder.join()
            images = list(recorder.folder.glob("*.png"))
            self.assertGreaterEqual(len(images), 2)
            self.assertEqual(
                len(images),
                len((recorder.folder / "manifest.jsonl").read_text().splitlines()),
            )
            self.assertEqual(recorder.error, "")

    def test_respects_max_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = SampleRecorder(Path(tmp), sample_fps=15, max_samples=1)
            rec.start()
            for _ in range(5):
                rec.submit(np.zeros((60, 90, 3), np.uint8))
                time.sleep(0.078)
            rec.stop()
            rec.join()
            self.assertEqual(rec.saved, 1)
            self.assertTrue(rec.limit_reached)


if __name__ == "__main__":
    unittest.main()
