"""Lightweight, non-verified evidence for color health-bar candidates."""
from __future__ import annotations

import cv2
import numpy as np


def evaluate_bar_evidence(
    hsv: np.ndarray, x: int, y: int, w: int, h: int,
    body_bgr: np.ndarray, scale: float,
) -> dict:
    """Score nearby container, glyphs and body detail; never infer exact HP."""
    height, width = hsv.shape[:2]
    margin = max(3, round(5 * scale))
    x1, x2 = max(0, x - margin), min(width, x + w + margin)
    y1, y2 = max(0, y - margin), min(height, y + h + margin)
    patch = hsv[y1:y2, x1:x2]
    if patch.size == 0:
        return {"confidence": 0.0, "body_support": 0.0}
    dark = float(np.mean(patch[:, :, 2] < 100))
    above = hsv[max(0, y - round(22 * scale)):y, x1:x2]
    glyphs = (
        float(np.mean((above[:, :, 1] < 90) & (above[:, :, 2] > 175)))
        if above.size else 0.0
    )
    body_support = 0.0
    if body_bgr.size and min(body_bgr.shape[:2]) >= 12:
        thumb = cv2.resize(body_bgr, (48, 48), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        edge_density = float(np.mean(cv2.Canny(gray, 45, 120) > 0))
        body_support = min(1.0, edge_density / 0.09)
    shape = min(1.0, (w / max(1, h)) / 12.0)
    confidence = min(
        0.88,
        0.35 + 0.23 * dark + 0.08 * shape
        + 0.12 * min(1.0, glyphs / 0.06)
        + 0.10 * body_support,
    )
    return {
        "confidence": round(confidence, 3),
        "body_support": round(body_support, 3),
        "container_dark_fraction": round(dark, 3),
        "glyph_support": round(glyphs, 4),
    }
