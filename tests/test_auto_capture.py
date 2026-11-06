"""Seleção automática segura da fonte, sem depender de Windows no CI."""

import unittest
from unittest.mock import patch

import numpy as np

from brawlla.capture import AutoWindowCapture, GameWindow
from brawlla.config import SessionOptions


class SourceTests(unittest.TestCase):
    def test_no_visible_game_window_reports_clear_error(self):
        with patch("brawlla.capture.list_game_windows", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "BlueStacks"):
                AutoWindowCapture()

    def test_uses_selected_window_and_closes_source(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        window = GameWindow(42, "Brawl Stars — BlueStacks", 640, 360)
        with (
            patch("brawlla.capture.list_game_windows", return_value=[window]),
            patch("brawlla.capture.WindowCapture") as source_cls,
        ):
            source = source_cls.return_value
            source.grab.return_value = frame
            source._last_rect = [50, 80, 640, 360]
            source.quality_flags = []
            auto = AutoWindowCapture()
            self.assertEqual(auto.source_label, window.title)
            self.assertIs(auto.grab(), frame)
            self.assertEqual(auto._last_rect, [50, 80, 640, 360])
            source_cls.assert_called_once_with(42)
            auto.close()
            source.close.assert_called_once()

    def test_session_auto_fps_validation(self):
        self.assertTrue(SessionOptions(auto_fps=True).auto_fps)
        with self.assertRaises(ValueError):
            SessionOptions(auto_fps=1)


if __name__ == "__main__":
    unittest.main()
