"""Preview substituível e eventos de controle preservados, entre threads."""

from collections import deque
from threading import Lock


class UiMailbox:
    def __init__(self):
        self._lock = Lock()
        self._preview = None
        self._events = deque()

    def offer(self, kind, payload):
        with self._lock:
            if kind == "frame":
                self._preview = (kind, payload)
            else:
                self._events.append((kind, payload))

    def drain(self):
        with self._lock:
            # Controles processados depois do preview; finalização não é perdida.
            items = ([self._preview] if self._preview is not None else []) + list(
                self._events
            )
            self._preview = None
            self._events.clear()
            return items

    def clear_preview(self):
        with self._lock:
            self._preview = None
