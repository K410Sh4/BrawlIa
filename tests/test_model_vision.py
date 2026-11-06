"""Testes sem rede, sem pesos de terceiros e sem dependência do ONNX Runtime."""

import unittest

import numpy as np

from brawlla.model_vision import ModelSchedule, decode_yolo, letterbox


class ModelVisionTests(unittest.TestCase):
    def test_schedule_never_represents_skipped_frames_as_observed(self):
        schedule = ModelSchedule(3)
        self.assertEqual(
            [i for i in range(10) if schedule.should_infer(i)], [0, 3, 6, 9]
        )
        with self.assertRaises(ValueError):
            ModelSchedule(0)
        with self.assertRaises(ValueError):
            schedule.should_infer(-1)

    def test_letterbox_keeps_aspect_and_maps_to_input(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        tensor, scale, pad = letterbox(frame, 640, 640)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(scale, 1)
        self.assertEqual(pad, (0, 140))

    def test_yolov8_common_transposed_export(self):
        # Uma classe, dois candidatos. Saída exportada (1, 5, 2).
        raw = np.array(
            [[[320, 320], [320, 320], [100, 100], [40, 40], [0.9, 0.1]]],
            dtype=np.float32,
        )
        # Alguns modelos com poucos candidatos não transpostos exigem formato normal;
        # aqui usamos 10 candidatos para exercitar o layout real CxN.
        raw = np.tile(raw, (1, 1, 5))
        raw[0, 4, 1:] = 0.1
        result = decode_yolo(raw, (640, 640, 3), 1, (0, 0), ["player"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["label"], "player")
        self.assertEqual(result[0]["xyxy"], [270.0, 300.0, 370.0, 340.0])

    def test_multiple_classes_and_class_aware_nms(self):
        # 2 classes = [x,y,w,h,score0,score1].
        data = np.array(
            [
                [200, 200, 40, 40, 0.95, 0.01],
                [200, 200, 40, 40, 0.90, 0.01],
                [200, 200, 40, 40, 0.01, 0.85],
            ],
            dtype=np.float32,
        )
        result = decode_yolo(data, (400, 400, 3), 1, (0, 0), ["player", "enemy"])
        self.assertEqual(len(result), 2)
        self.assertEqual({r["label"] for r in result}, {"player", "enemy"})

    def test_xyxy_end_to_end_and_letterbox_coordinates(self):
        data = np.array([[40, 150, 140, 250, 0.95, 0]], dtype=np.float32)
        result = decode_yolo(
            data,
            (360, 640, 3),
            1,
            (0, 140),
            ["enemy"],
            export_format="xyxy",
        )
        self.assertEqual(result[0]["xyxy"], [40.0, 10.0, 140.0, 110.0])

    def test_validation_and_empty_results(self):
        with self.assertRaises(ValueError):
            decode_yolo(np.zeros((4, 4, 4)), (100, 100, 3), 1, (0, 0))
        self.assertEqual(decode_yolo(np.zeros((20, 5)), (100, 100, 3), 1, (0, 0)), [])


if __name__ == "__main__":
    unittest.main()
