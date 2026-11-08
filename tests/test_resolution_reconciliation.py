"""Regression tests for resolution reconciliation in capture fallback."""

import unittest
from unittest.mock import Mock

from brawlla.recognition_v1.runtime import reconcile_screen_size


class ResolutionReconciliationTests(unittest.TestCase):
    def test_first_frame_does_not_reset_memory(self):
        memory = Mock()
        result = reconcile_screen_size(None, (640, 360), memory)
        self.assertEqual(result, (640, 360))
        memory.reset.assert_not_called()

    def test_unchanged_resolution_preserves_tracks(self):
        memory = Mock()
        result = reconcile_screen_size((640, 360), (640, 360), memory)
        self.assertEqual(result, (640, 360))
        memory.reset.assert_not_called()

    def test_resolution_change_resets_tracks(self):
        memory = Mock()
        result = reconcile_screen_size((640, 360), (1280, 720), memory)
        self.assertEqual(result, (1280, 720))
        memory.reset.assert_called_once_with("resolucao_alterada")


if __name__ == "__main__":
    unittest.main()
