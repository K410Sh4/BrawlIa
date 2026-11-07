"""Captura de imagem local, SEM enviar comandos de jogo.

Backend rápido: pixels da janela do BlueStacks via MSS. O cliente deve manter
essa janela visível (MSS lê o desktop: outras janelas cobrindo o jogo aparecem).
Backend de compatibilidade: screencap ADB, lento, sem recapturar conexão por frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class GameWindow:
    handle: int
    title: str
    width: int
    height: int


class CaptureSource(Protocol):
    def grab(self) -> np.ndarray: ...
    def close(self) -> None: ...


def enable_dpi_awareness() -> None:
    """Sincroniza coordenadas do Win32 com pixels físicos (DPI 125%/150%)."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI-aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def list_game_windows() -> list[GameWindow]:
    """Enumera janelas candidatas sem pacotes win32 extras (somente Windows)."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    found: list[GameWindow] = []

    @callback_type
    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length < 1:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if not any(token in title.lower() for token in ("bluestacks", "brawl stars")):
            return True
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return True
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width >= 320 and height >= 240:
            found.append(GameWindow(int(hwnd), title, width, height))
        return True

    user32.EnumWindows(callback, 0)
    return sorted(
        found,
        key=lambda w: ("brawl stars" not in w.title.lower(), -(w.width * w.height)),
    )


def _window_client_bbox(handle: int) -> tuple[int, int, int, int]:
    if sys.platform != "win32":
        raise RuntimeError("Captura de janela é disponível somente no Windows.")
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    hwnd = wintypes.HWND(handle)
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    if not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        raise RuntimeError("BlueStacks fechado ou minimizado; reabra a janela.")
    rect = wintypes.RECT()
    origin = wintypes.POINT(0, 0)
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)) or not user32.ClientToScreen(
        hwnd, ctypes.byref(origin)
    ):
        raise RuntimeError("Não foi possível localizar a área do jogo.")
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w < 320 or h < 240:
        raise RuntimeError("Janela do BlueStacks está pequena demais.")
    return origin.x, origin.y, w, h


def _window_occlusion_flags(handle, rect):
    """Teste pontual: outras janelas sobre 3 pontos do cliente. Não é prova de visibilidade total."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.WindowFromPoint.argtypes = [wintypes.POINT]
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    root = user32.GetAncestor(wintypes.HWND(handle), 2) or handle
    x, y, w, h = rect
    for fx, fy in ((0.15, 0.15), (0.5, 0.5), (0.85, 0.85)):
        hit = user32.WindowFromPoint(
            wintypes.POINT(round(x + w * fx), round(y + h * fy))
        )
        if hit and (user32.GetAncestor(hit, 2) or hit) != root:
            return ["possible_window_occlusion"]
    return []


class WindowCapture:
    """Captura pixels visíveis do cliente da janela, sem leitura ADB."""

    def __init__(self, handle: int):
        if sys.platform != "win32":
            raise RuntimeError("Modo rápido exige Windows.")
        try:
            import mss
        except ImportError as exc:
            raise RuntimeError(
                "Biblioteca mss ausente. Instale as dependências."
            ) from exc
        self.handle = int(handle)
        # cvtColor cria um ndarray independente em cada grab().
        self.fresh_frames = True
        self._mss = mss.mss()  # Criar e consumir no MESMO thread (recursos GDI).
        self._last_rect = None
        self.quality_flags = []
        self._next_visibility_check = 0

    def grab(self) -> np.ndarray:
        x, y, w, h = _window_client_bbox(self.handle)
        bounds = self._mss.monitors[0]  # desktop virtual, inclui monitores negativos
        left = max(x, bounds["left"])
        top = max(y, bounds["top"])
        right = min(x + w, bounds["left"] + bounds["width"])
        bottom = min(y + h, bounds["top"] + bounds["height"])
        if (left, top, right, bottom) != (x, y, x + w, y + h):
            raise RuntimeError(
                "Janela parcialmente fora da tela; reposicione antes de capturar."
            )
        if right - left < 320 or bottom - top < 240:
            raise RuntimeError("A janela precisa estar visível dentro do monitor.")
        self._last_rect = [x, y, w, h]
        if time.monotonic() >= self._next_visibility_check:
            self._next_visibility_check = time.monotonic() + 0.5
            self.quality_flags = _window_occlusion_flags(self.handle, self._last_rect)
        shot = np.asarray(
            self._mss.grab(
                {
                    "left": left,
                    "top": top,
                    "width": right - left,
                    "height": bottom - top,
                }
            )
        )
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

    def grab_region(self, rect: list[int] | tuple[int, int, int, int]) -> np.ndarray:
        """Captura SOMENTE o retângulo xywh relativo ao cliente da janela.

        O objeto MSS permanece no thread de captura; não efetua leitura completa.
        """
        wx, wy, ww, wh = _window_client_bbox(self.handle)
        if len(rect) != 4 or any(type(v) is not int for v in rect):
            raise ValueError("Região deve ser xywh de inteiros")
        rx, ry, rw, rh = rect
        if rw <= 0 or rh <= 0 or rx < 0 or ry < 0 or rx + rw > ww or ry + rh > wh:
            raise ValueError("Região está fora do cliente do BlueStacks")
        bounds = self._mss.monitors[0]
        if (
            wx < bounds["left"]
            or wy < bounds["top"]
            or wx + ww > bounds["left"] + bounds["width"]
            or wy + wh > bounds["top"] + bounds["height"]
        ):
            raise RuntimeError("Janela parcialmente fora da tela")
        self._last_rect = [wx, wy, ww, wh]
        if time.monotonic() >= self._next_visibility_check:
            self._next_visibility_check = time.monotonic() + 0.5
            self.quality_flags = _window_occlusion_flags(self.handle, self._last_rect)
        shot = np.asarray(
            self._mss.grab(
                {
                    "left": wx + rx,
                    "top": wy + ry,
                    "width": rw,
                    "height": rh,
                }
            )
        )
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self._mss.close()


class DxcamWindowCapture:
    """Desktop Duplication opcional. Janela precisa estar visível no monitor.

    Requer pacote dxcam no Windows; usa cópias BGR independentes para que
    o buffer não seja sobrescrito enquanto a análise lê o quadro.
    """

    def __init__(self, handle):
        if sys.platform != "win32":
            raise RuntimeError("DXcam exige Windows")
        import dxcam

        self.handle = int(handle)
        self._camera = dxcam.create(output_color="BGR")
        if self._camera is None:
            raise RuntimeError("DXcam não conseguiu iniciar a captura")
        self.fresh_frames = True
        self.quality_flags = []
        self._last_rect = None
        self._next_visibility_check = 0.0

    def grab(self):
        x, y, width, height = _window_client_bbox(self.handle)
        # O backend padrão captura o monitor primário. Coordenadas negativas
        # ou áreas fora dele devem recair no MSS, sem entregar imagem errada.
        region = (x, y, x + width, y + height)
        if x < 0 or y < 0:
            raise RuntimeError("DXcam primário não cobre este monitor")
        frame = self._camera.grab(region=region, new_frame_only=False)
        if (
            frame is None
            or frame.ndim != 3
            or frame.shape[:2] != (height, width)
            or frame.shape[2] != 3
        ):
            raise RuntimeError("DXcam sem quadro completo da janela")
        self._last_rect = [x, y, width, height]
        if time.monotonic() >= self._next_visibility_check:
            self._next_visibility_check = time.monotonic() + 0.5
            self.quality_flags = _window_occlusion_flags(self.handle, self._last_rect)
        return frame

    def grab_region(self, rect: list[int] | tuple[int, int, int, int]) -> np.ndarray:
        """Desktop Duplication de uma região, com coordenadas globais verificadas."""
        wx, wy, ww, wh = _window_client_bbox(self.handle)
        if len(rect) != 4 or any(type(v) is not int for v in rect):
            raise ValueError("Região deve ser xywh de inteiros")
        rx, ry, rw, rh = rect
        if rw <= 0 or rh <= 0 or rx < 0 or ry < 0 or rx + rw > ww or ry + rh > wh:
            raise ValueError("Região está fora do cliente do BlueStacks")
        if wx < 0 or wy < 0:
            raise RuntimeError("DXcam primário não cobre este monitor")
        frame = self._camera.grab(
            region=(wx + rx, wy + ry, wx + rx + rw, wy + ry + rh),
            new_frame_only=False,
        )
        if frame is None or frame.shape != (rh, rw, 3):
            raise RuntimeError("DXcam retornou uma região incompleta")
        self._last_rect = [wx, wy, ww, wh]
        if time.monotonic() >= self._next_visibility_check:
            self._next_visibility_check = time.monotonic() + 0.5
            self.quality_flags = _window_occlusion_flags(self.handle, self._last_rect)
        return frame.copy()

    def close(self):
        self._camera.release()


class AutoWindowCapture:
    """Escolhe a melhor janela visível automaticamente, no thread de captura.

    Não troca de janela silenciosamente no meio da sessão.
    """

    def __init__(self):
        windows = list_game_windows()
        if not windows:
            raise RuntimeError(
                "Nenhuma janela BlueStacks/Brawl Stars visível. "
                "Abra o emulador na tela e tente novamente."
            )
        chosen = windows[0]
        self.source_label = chosen.title
        self._backend = "mss"
        preference = os.getenv("BRAWLIA_CAPTURE_BACKEND", "mss").lower()
        if preference not in ("mss", "dxcam"):
            raise ValueError("BRAWLIA_CAPTURE_BACKEND deve ser mss ou dxcam")
        if preference == "dxcam":
            try:
                self._source = DxcamWindowCapture(chosen.handle)
                self._backend = "dxcam"
            except (RuntimeError, ImportError, OSError, ValueError):
                self._source = WindowCapture(chosen.handle)
        else:
            self._source = WindowCapture(chosen.handle)
        self.backend = self._backend
        self.fresh_frames = True
        self.handle = chosen.handle
        self._last_rect = None
        self.quality_flags = []

    def grab(self):
        try:
            result = self._source.grab()
        except (RuntimeError, OSError, ValueError):
            if self._backend != "dxcam":
                raise
            self._source.close()
            self._source = WindowCapture(self.handle)
            self._backend = "mss"
            self.backend = self._backend
            result = self._source.grab()
        self._last_rect = self._source._last_rect
        self.quality_flags = self._source.quality_flags
        return result

    def grab_region(self, rect: list[int] | tuple[int, int, int, int]) -> np.ndarray:
        """Captura região e volta ao MSS se DXcam falhar durante a sessão."""
        try:
            result = self._source.grab_region(rect)
        except (RuntimeError, OSError, ValueError):
            if self._backend != "dxcam":
                raise
            self._source.close()
            self._source = WindowCapture(self.handle)
            self._backend = "mss"
            self.backend = self._backend
            result = self._source.grab_region(rect)
        self._last_rect = self._source._last_rect
        self.quality_flags = self._source.quality_flags
        return result

    def close(self):
        self._source.close()


def find_adb() -> str:
    candidates = [
        os.environ.get("BRAWLLA_ADB", ""),
        shutil.which("adb") or "",
        shutil.which("HD-Adb.exe") or "",
        r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
        r"C:\Program Files (x86)\BlueStacks_nxt\HD-Adb.exe",
        r"C:\Program Files\Android\platform-tools\adb.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("ADB não encontrado. Configure BRAWLLA_ADB com HD-Adb.exe.")


class AdbCapture:
    """Fallback: conecta só ao iniciar. 30 FPS não são garantidos nesse backend."""

    def __init__(self, port: int = 5555):
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("Porta ADB inválida.")
        self.port = port
        self.adb = find_adb()
        self.host = f"127.0.0.1:{port}"
        self.flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            [self.adb, "connect", self.host],
            capture_output=True,
            timeout=7,
            creationflags=self.flags,
        )
        if result.returncode != 0 or b"failed" in result.stdout.lower():
            msg = (result.stderr or result.stdout).decode("utf-8", "replace")
            raise RuntimeError("ADB não conectado: " + msg[:250])

    def grab(self) -> np.ndarray:
        result = subprocess.run(
            [self.adb, "-s", self.host, "exec-out", "screencap", "-p"],
            capture_output=True,
            timeout=10,
            creationflags=self.flags,
        )
        if result.returncode != 0 or not result.stdout:
            raise RuntimeError(
                (result.stderr or b"Falha ADB").decode("utf-8", "replace")[:300]
            )
        frame = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("ADB não retornou imagem PNG válida.")
        return frame

    def close(self) -> None:
        pass


def grab_blue_stacks(port: int = 5555) -> np.ndarray:
    """Compatibilidade com V0.1: captura isolada, não para loop em FPS alto."""
    capture = AdbCapture(port)
    try:
        return capture.grab()
    finally:
        capture.close()
