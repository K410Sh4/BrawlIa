import unittest

import cv2
import numpy as np
from brawlla.vision import analyze


class VisionTests(unittest.TestCase):
    def test_synthetic_hp_bars(self):
        frame = np.zeros((600, 1000, 3), dtype=np.uint8)
        # Saturação/valor coerentes com as barras usadas na heurística.
        name_width = cv2.getTextSize("MEU PLAYER", cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[
            0
        ][0]
        cv2.putText(
            frame,
            "MEU PLAYER",
            (460 - name_width // 2, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (30, 230, 50),
            2,
        )
        cv2.rectangle(frame, (420, 220), (499, 229), (35, 220, 70), -1)  # verde BGR
        cv2.rectangle(frame, (520, 300), (605, 310), (50, 40, 232), -1)  # vermelho BGR
        cv2.putText(
            frame,
            "7560",
            (432, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
        )
        # Glifos claros acima distinguem a barra de tijolos e outros retângulos do cenário.
        cv2.putText(
            frame,
            "3500",
            (530, 295),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
        )
        out, detected = analyze(frame)
        self.assertEqual(out.shape, frame.shape)
        self.assertIsNotNone(detected["player_health_bar_candidate"])
        self.assertGreaterEqual(len(detected["red_health_bar_candidates"]), 1)

    def test_small_image_rejected(self):
        with self.assertRaises(ValueError):
            analyze(np.zeros((50, 50, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
