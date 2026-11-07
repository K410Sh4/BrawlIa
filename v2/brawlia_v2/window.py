"""Windows-only window discovery without third-party GUI automation.

The capture source reads visible desktop pixels. It never sends game input.
"""

from dataclasses import dataclass
import sys


@dataclass(frozen=True, slots=True)
class WindowRect:
    handle: int
    title: str
    left: int
    top: int
    width: int
    height: int


def enable_dpi_awareness() -> None:
    """Use physical pixels so the Win32 client bounds match MSS coordinates."""
    if sys.platform != "win32":
        return
    import ctypes

    user32 = ctypes.windll.user32
    try:
        # PER_MONITOR_AWARE_V2. Windows 10+; negative HANDLE represented as int.
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def find_bluestacks_window() -> WindowRect:
    """Find an unminimized BlueStacks window; prioritize an explicit game title.

    Window placement is refreshed during capture because the user may resize
    or move the emulator. No hidden-window or background-device capture.
    """
    if sys.platform != "win32":
        raise RuntimeError("A captura de janela BlueStacks exige Windows")

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL

    found = []

    @callback_type
    def visit(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        size = user32.GetWindowTextLengthW(hwnd)
        if not size:
            return True
        title_buf = ctypes.create_unicode_buffer(size + 1)
        user32.GetWindowTextW(hwnd, title_buf, size + 1)
        title = title_buf.value
        title_lower = title.casefold()
        if "bluestacks" not in title_lower and "brawl stars" not in title_lower:
            return True
        rect = wintypes.RECT()
        point = wintypes.POINT(0, 0)
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return True
        if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
            return True
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width < 300 or height < 200:
            return True
        found.append(WindowRect(int(hwnd), title, point.x, point.y, width, height))
        return True

    if not user32.EnumWindows(visit, 0):
        raise OSError(f"Não foi possível enumerar janelas: {ctypes.get_last_error()}")
    if not found:
        raise RuntimeError("Abra o BlueStacks com Brawl Stars visível e não minimizado")
    return max(found, key=lambda item: ("brawl stars" in item.title.casefold(), item.width * item.height))
