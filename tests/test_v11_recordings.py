"""Regressões de navegação para gravações e política de cópias da captura."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from brawlla import recordings
from brawlla.dataset import crop_frame


class RecordingFolderTests(unittest.TestCase):
    def test_open_root_before_first_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(recordings.recording_folder(root), root / "gravacoes")

    def test_latest_session_after_relaunch_and_current_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            past = root / "gravacoes" / "20261105_100000_aaaaaa"
            recent = root / "gravacoes" / "20261106_200000_bbbbbb"
            for folder in (past, recent):
                folder.mkdir(parents=True)
                (folder / "sessao.json").write_text("{}", encoding="utf-8")
            self.assertEqual(recordings.recording_folder(root), recent)
            self.assertEqual(recordings.recording_folder(root, past), past)
            with self.assertRaises(ValueError):
                recordings.recording_folder(root, root / "other")

    def test_ignores_non_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            empty = root / "gravacoes" / "20261106_999999_empty"
            valid = root / "gravacoes" / "20261106_100000_valid"
            empty.mkdir(parents=True)
            valid.mkdir(parents=True)
            (valid / "sessao.json").touch()
            self.assertEqual(recordings.recording_folder(root), valid)

    def test_opens_folder_without_shell_in_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with (
                patch.object(recordings.sys, "platform", "win32"),
                patch.object(recordings.os, "startfile", create=True) as start,
            ):
                recordings.open_recording_folder(folder)
                start.assert_called_once_with(str(folder.resolve()))

    def test_missing_folder_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                recordings.open_recording_folder(Path(directory) / "missing")


class CaptureCopyTests(unittest.TestCase):
    def test_full_image_can_use_fresh_source_buffer(self):
        original = np.zeros((720, 1280, 3), dtype=np.uint8)
        copied, _ = crop_frame(original)
        shared, _ = crop_frame(original, copy_full_frame=False)
        self.assertIsNot(original, copied)
        self.assertIs(original, shared)

    def test_cropped_image_is_independent_even_with_fresh_source(self):
        original = np.zeros((720, 1280, 3), dtype=np.uint8)
        cropped, metadata = crop_frame(
            original, (0.1, 0.1, 0.5, 0.5), copy_full_frame=False
        )
        self.assertEqual(metadata["roi_xywh"][2:], [640, 360])
        cropped[0, 0] = [255, 255, 255]
        self.assertEqual(int(original.sum()), 0)


if __name__ == "__main__":
    unittest.main()
