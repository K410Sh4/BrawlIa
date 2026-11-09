"""Regression coverage for runtime resource finalization."""

import unittest
from unittest.mock import Mock, patch

from brawlla.recognition_v1.runtime import finalize_runtime_resources


class RuntimeCleanupTests(unittest.TestCase):
    def test_capture_close_failure_still_stops_and_joins_store(self):
        capture = Mock()
        capture.close.side_effect = OSError("capture close failed")
        store = Mock()
        errors = finalize_runtime_resources(capture, store, preview=False)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], OSError)
        store.stop.assert_called_once_with()
        store.join.assert_called_once_with()

    def test_preview_failure_still_stops_and_joins_store(self):
        store = Mock()
        with patch(
            "brawlla.recognition_v1.runtime.cv2.destroyAllWindows",
            side_effect=RuntimeError("preview close failed"),
        ):
            errors = finalize_runtime_resources(None, store, preview=True)
        self.assertEqual(len(errors), 1)
        store.stop.assert_called_once_with()
        store.join.assert_called_once_with()

    def test_stop_failure_still_joins_store(self):
        store = Mock()
        store.stop.side_effect = RuntimeError("stop failed")
        errors = finalize_runtime_resources(None, store, preview=False)
        self.assertEqual(len(errors), 1)
        store.join.assert_called_once_with()

    def test_normal_cleanup(self):
        capture = Mock()
        store = Mock()
        self.assertEqual(finalize_runtime_resources(capture, store, preview=False), [])
        capture.close.assert_called_once_with()
        store.stop.assert_called_once_with()
        store.join.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
