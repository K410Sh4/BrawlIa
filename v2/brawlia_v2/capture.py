"""Fast visible-pixel capture. Capture and image interpretation remain separate."""

import time

import numpy as np

from .window import find_bluestacks_window


class MSSBlueStacksSource:
    """One MSS instance, owned by one capture thread.

    Returned arrays reference *new* screenshot buffers and are read-only.
    They remain alive until the consumer discards the frame. No PNG/MP4.
    """

    def __init__(self, locator=find_bluestacks_window, rect_refresh_s=0.5):
        self.locator = locator
        self.rect_refresh_s = rect_refresh_s
        self._mss = None
        self._rect = None
        self._refreshed_at = float("-inf")

    def open(self):
        import mss

        self._mss = mss.mss()
        self._rect = None
        return self

    def grab(self):
        if self._mss is None:
            raise RuntimeError("Fonte de captura não inicializada")
        now = time.perf_counter()
        if self._rect is None or now - self._refreshed_at >= self.rect_refresh_s:
            self._rect = self.locator()
            self._refreshed_at = now
        r = self._rect
        monitor = self._mss.monitors[0]
        if (
            r.left < monitor["left"]
            or r.top < monitor["top"]
            or r.left + r.width > monitor["left"] + monitor["width"]
            or r.top + r.height > monitor["top"] + monitor["height"]
        ):
            raise RuntimeError("Janela fora da área visível; mova o BlueStacks inteiramente para o monitor")
        shot = self._mss.grab(
            {"left": r.left, "top": r.top, "width": r.width, "height": r.height}
        )
        # Use the screenshot's own freshly allocated bytearray, without a full-frame
        # numpy copy or BGRA->BGR conversion. np.frombuffer owns a reference.
        pixels = np.frombuffer(shot.raw, dtype=np.uint8).reshape(r.height, r.width, 4)
        pixels.flags.writeable = False
        return pixels

    def close(self):
        if self._mss is not None:
            self._mss.close()
            self._mss = None


class SyntheticSource:
    """Low-cost, deterministic visual source for cross-platform smoke tests."""

    def __init__(self, width=1280, height=720):
        self.pixels = np.zeros((height, width, 4), dtype=np.uint8)
        self.pixels[:, :, 3] = 255
        self.pixels.flags.writeable = False

    def open(self):
        return self

    def grab(self):
        return self.pixels

    def close(self):
        pass
