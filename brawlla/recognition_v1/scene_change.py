"""Detecção leve de mudanças de tela em regiões realmente adquiridas.

O monitor apenas solicita uma nova captura completa: ele NÃO atribui estados
a partir de diferenças de pixels nem considera regiões não observadas.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import Rect

TERMINAL_STATES = frozenset(("derrota", "resultados"))


def scene_probe_rect(size: tuple[int, int]) -> Rect:
    w, h = size
    return Rect(round(w * 0.35), round(h * 0.18),
                round(w * 0.30), round(h * 0.32))


def scene_signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (40, 32), interpolation=cv2.INTER_AREA)


def detect_scene_transition(reference: np.ndarray, current: np.ndarray,
                            threshold: float = 0.27) -> bool:
    if reference.shape != current.shape or reference.size == 0:
        return False
    difference = float(np.mean(cv2.absdiff(reference, current)) / 255.0)
    return difference >= threshold


class SceneTransitionMonitor:
    def __init__(self, change_votes: int = 2, threshold: float = 0.27,
                 cooldown_s: float = 0.7):
        self.change_votes = change_votes
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.reference: np.ndarray | None = None
        self.reference_state: str | None = None
        self.streak = 0
        self.last_request = float("-inf")

    @staticmethod
    def _extract(regions, size: tuple[int, int]) -> np.ndarray | None:
        probe = scene_probe_rect(size)
        for region in regions:
            if region.rect.intersection_area(probe) != probe.area:
                continue
            x = probe.x - region.rect.x
            y = probe.y - region.rect.y
            return scene_signature(
                region.image[y:y + probe.h, x:x + probe.w]
            )
        return None

    def observe(self, regions, size: tuple[int, int], state: str,
                now: float, *, full: bool) -> bool:
        if state not in TERMINAL_STATES:
            self.reference = None
            self.reference_state = None
            self.streak = 0
            return False
        current = self._extract(regions, size)
        if current is None:
            return False
        if self.reference is None or self.reference_state != state:
            self.reference = current
            self.reference_state = state
            self.streak = 0
            return False
        if full:
            # Recalibrar somente depois de uma varredura completa confirmada.
            self.reference = current
            self.streak = 0
            return False
        if detect_scene_transition(self.reference, current, self.threshold):
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.change_votes and (
            now - self.last_request >= self.cooldown_s
        ):
            self.streak = 0
            self.last_request = now
            return True
        return False
