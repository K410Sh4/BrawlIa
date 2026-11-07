"""Âncoras inferiores do HUD (candidatas geométricas) + memória de combate.

Círculos não confirmam botões isoladamente: falsos positivos são esperados;
o StateDetector existente confirma menus, combate e resultados no quadro completo.
"""

from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class HUDAnchors:
    attack: bool = False
    movement: bool = False
    gadget: bool = False

    @property
    def score(self) -> float:
        return (
            (0.45 if self.attack else 0.0)
            + (0.35 if self.movement else 0.0)
            + (0.20 if self.gadget else 0.0)
        )

    @property
    def combat_candidate(self) -> bool:
        return self.attack and self.movement


def detect_controls(
    frame: np.ndarray, origin: tuple[int, int], screen_size: tuple[int, int]
) -> HUDAnchors:
    """Procura candidatos circulares nas regiões inferiores da tela."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return HUDAnchors()
    sw, sh = screen_size
    if sw < 320 or sh < 200:
        return HUDAnchors()
    h, w = frame.shape[:2]
    if h < 50 or w < 50:
        return HUDAnchors()
    # Hough works on a small control patch, never on full-screen quadrants.
    # Limit the input resolution so large BlueStacks windows remain affordable.
    scale = min(1.0, 256 / max(w, h))
    reduced = (
        cv2.resize(
            frame,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else frame
    )
    gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    min_r = max(6, round(sw * 0.012 * scale))
    max_r = max(min_r + 2, round(sw * 0.073 * scale))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.4,
        minDist=max(12, min_r * 2),
        param1=100,
        param2=29,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return HUDAnchors()
    left = False
    right = []
    ox, oy = origin
    for cx, cy, radius in np.round(circles[0]).astype(int):
        gx, gy = ox + round(cx / scale), oy + round(cy / scale)
        if gy < sh * 0.60:
            continue
        if gx < sw * 0.40:
            left = True
        elif gx > sw * 0.60:
            right.append((gx, gy, radius))
    attack = bool(right)
    # Acessório opcional: segundo círculo menor e distinto, não exigido para combate.
    gadget = any(
        abs(a[0] - b[0]) > min_r * 2 and min(a[2], b[2]) <= max(a[2], b[2]) * 0.80
        for i, a in enumerate(right)
        for b in right[i + 1 :]
    )
    return HUDAnchors(attack=attack, movement=left, gadget=gadget)


class HUDContext:
    """Estado global não depende de uma barra ou de um único frame."""

    def __init__(self, grace_s: float = 2.5):
        self.grace_s = grace_s
        self.state = "desconhecido"
        self.last_combat = None
        self.last_confirmed = None
        self.anchor_streak = 0
        self.confidence = 0.0

    def update(
        self,
        anchors: HUDAnchors,
        now: float,
        visual_state: str | None = None,
        visual_confidence: float = 0.0,
    ) -> dict:
        if anchors.combat_candidate:
            self.anchor_streak += 1
        else:
            self.anchor_streak = 0
        if visual_state == "combate":
            self.state = "combate"
            self.confidence = max(0.5, float(visual_confidence))
            self.last_confirmed = now
            self.last_combat = now
            reason = "state_detector"
        elif visual_state in ("pre_partida", "derrota", "resultados"):
            self.state = visual_state
            self.confidence = max(0.5, float(visual_confidence))
            self.last_confirmed = now
            reason = "transicao_confirmada"
        elif self.anchor_streak >= 2:
            self.state = "combate"
            self.last_combat = now
            self.confidence = min(0.75, 0.45 + anchors.score * 0.30)
            reason = "ancoras_circulares_heuristicas"
        elif (
            self.state == "combate"
            and self.last_combat is not None
            and (now - self.last_combat <= self.grace_s)
        ):
            self.confidence = min(self.confidence, 0.49)
            reason = "hud_temporariamente_ausente"
        else:
            self.state = "desconhecido"
            self.confidence = 0.0
            reason = "sem_evidencia_atual"
        return {
            "name": self.state,
            "confidence": round(self.confidence, 3),
            "uncertain": reason
            in (
                "hud_temporariamente_ausente",
                "ancoras_circulares_heuristicas",
                "sem_evidencia_atual",
            ),
            "reason": reason,
            "anchors": {
                "attack": anchors.attack,
                "movement": anchors.movement,
                "gadget": anchors.gadget,
                "score": anchors.score,
            },
        }
