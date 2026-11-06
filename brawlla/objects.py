"""Triagem conservadora de barras vermelhas: caixas possíveis vs outros alvos.

Não é um classificador treinado. Evidência insuficiente = indefinido.
"""

from __future__ import annotations
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
from .config import VisionConfig

BOX_PATH = Path(__file__).resolve().parent / "assets" / "caixa_exemplo.png"


class ObjectClassifier:
    def __init__(self, config=None) -> None:
        self.config = config or VisionConfig()
        example = cv2.imread(str(BOX_PATH), cv2.IMREAD_GRAYSCALE)
        if example is None:
            raise RuntimeError("Exemplo de caixa ausente")
        self.box = example

    @lru_cache(maxsize=16)
    def _scaled_templates(self, frame_width):
        """Evita cinco redimensionamentos por candidato a cada quadro."""
        templates = []
        for s in (0.60, 0.72, 0.84, 0.96, 1.10):
            factor = s * (frame_width / self.config.reference_width)
            sw = max(8, round(self.box.shape[1] * factor))
            sh = max(8, round(self.box.shape[0] * factor))
            templates.append(
                (sw, sh, cv2.resize(self.box, (sw, sh), interpolation=cv2.INTER_AREA))
            )
        return templates

    def classify(self, frame: np.ndarray, health_bar: dict) -> dict:
        h, w = frame.shape[:2]
        x = int(health_bar["x"])
        y = int(health_bar["y"])
        bw = int(health_bar["width"])
        bh = int(health_bar["height"])
        scale = w / self.config.legacy_reference_width
        pad = round(26 * scale)
        y1 = min(h, y + bh + round(5 * scale))
        y2 = min(h, y + bh + round(105 * scale))
        x1, x2 = max(0, x - pad), min(w, x + bw + pad)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0 or min(patch.shape[:2]) < 20:
            return {"kind": "indefinido", "confidence": 0.0, "box_evidence": 0.0}
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Textura/forma da caixa, com quatro tamanhos para diferentes resoluções.
        max_match = 0.0
        for sw, sh, small in self._scaled_templates(w):
            if sw >= gray.shape[1] or sh >= gray.shape[0]:
                continue
            max_match = max(
                max_match,
                float(
                    cv2.minMaxLoc(cv2.matchTemplate(gray, small, cv2.TM_CCOEFF_NORMED))[
                        1
                    ]
                ),
            )
        # Confirmar evidência de madeira/laranja abaixo da barra, evitando
        # confundir brawlers amarelos com caixas. Ainda é APENAS suspeita.
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        ratio_scale = max(0.4, w / self.config.reference_width)
        crop_y1 = max(0, round(3 * ratio_scale))
        crop_y2 = min(hsv.shape[0], round(72 * ratio_scale))
        crop_x1 = max(0, x - x1)
        crop_x2 = min(hsv.shape[1], x + bw + round(5 * ratio_scale) - x1)
        tight = hsv[crop_y1:crop_y2, crop_x1:crop_x2]
        if tight.size:
            hh, ss, vv = cv2.split(tight)
            brown = float(
                np.mean((hh >= 3) & (hh <= 24) & (ss >= 100) & (vv >= 45) & (vv <= 170))
            )
            orange = float(np.mean((hh >= 3) & (hh <= 30) & (ss > 100) & (vv > 60)))
        else:
            brown = orange = 0.0
        if max_match >= 0.44 and brown >= 0.12 and orange >= 0.30:
            kind = "caixa_provavel"
            confidence = min(0.82, 0.52 + 0.7 * brown + 0.10 * max_match)
        elif max_match <= 0.33 and brown < 0.12:
            kind = "personagem_possivel"
            confidence = min(0.65, 0.35 + (0.33 - max_match))
        else:
            kind = "indefinido"
            confidence = 0.25
        return {
            "kind": kind,
            "confidence": round(confidence, 3),
            "box_evidence": round(max_match, 3),
            "brown_fraction": round(brown, 3),
        }
