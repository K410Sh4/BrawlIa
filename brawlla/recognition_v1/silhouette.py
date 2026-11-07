"""Assinatura estrutural heurística, NÃO reconhecimento treinado de brawlers."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Signature:
    edges: np.ndarray
    histogram: np.ndarray


def extract_silhouette(frame: np.ndarray) -> Signature | None:
    """Extrai assinatura leve; o topo variável (texto/vida) é ignorado."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return None
    h, w = frame.shape[:2]
    if h < 20 or w < 16:
        return None
    body = frame[max(1, round(h * 0.24)):, :]
    small = cv2.resize(body, (48, 48), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 115)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram)
    return Signature(edges, histogram)


def compare_silhouettes(previous: Signature | None, current: Signature | None) -> float | None:
    """Similaridade 0..1; indício auxiliar, nunca identidade confirmada."""
    if previous is None or current is None:
        return None
    a = previous.edges > 0
    b = current.edges > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    shape = inter / union if union else 1.0
    color = cv2.compareHist(previous.histogram, current.histogram, cv2.HISTCMP_CORREL)
    return float(np.clip(0.65 * shape + 0.35 * max(0.0, color), 0.0, 1.0))


def structural_change_ratio(previous: np.ndarray, current: np.ndarray,
                            stable_mask: np.ndarray, threshold: int = 25) -> float | None:
    """Muda somente os pixels relevantes; não compara números/HP se mascarados."""
    if previous.shape != current.shape or previous.ndim != 3:
        return None
    if stable_mask.shape != previous.shape[:2]:
        raise ValueError("Máscara estrutural com tamanho incorreto")
    relevant = stable_mask.astype(bool)
    if not relevant.any():
        return None
    a = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(a, b)[relevant] > threshold))
