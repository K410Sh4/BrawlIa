"""Identificação temporal de nome + números + recipiente da barra do jogador."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
from .config import VisionConfig
from .domain import Timeline
from .health import bar_observations


def white_number_support(hsv, x, y, width, config=None):
    """Compatibilidade: fração de pixels claros; não é OCR nem identidade."""
    config = config or VisionConfig()
    scale = hsv.shape[1] / config.reference_width
    patch = hsv[
        max(0, y - round(18 * scale)) : max(0, y),
        max(0, x - round(2 * scale)) : min(hsv.shape[1], x + width + round(3 * scale)),
    ]
    if not patch.size:
        return 0.0
    white = cv2.inRange(
        patch,
        (0, 0, 185 if scale >= 0.75 else 165),
        (179, 60 if scale >= 0.75 else 80, 255),
    )
    count = int(np.count_nonzero(white))
    return (
        float(count / white.size) if count >= max(5, round(12 * scale * scale)) else 0.0
    )


class PlayerLocator:
    def __init__(self, config=None):
        self.config = config or VisionConfig()
        self.clock = Timeline()
        self.reset()

    def reset(self):
        self.last_seen = self.previous = self.resolution = None
        self.missed = 0
        self.reference_width = None
        self.reference_name = None
        self.name_visual_key = None
        self.velocity = np.zeros(2)
        self.last_global_at = None
        self.diagnostics = {}

    def _names(self, hsv, evidence=None, region=None, *, highlight=False):
        """Propõe faixas de glifos; a barra abaixo ainda deve confirmar cada nome.

        Em vegetação, o verde amplo une letras e arbustos em um só contorno.
        O fallback utiliza somente tons mais luminosos do mesmo intervalo HSV.
        Ele é acionado apenas quando a via normal não encontrou jogador.
        """
        h, w = hsv.shape[:2]
        scale = w / self.config.reference_width
        green = (
            evidence.color_masks[0][0]
            if evidence
            else cv2.inRange(hsv, self.config.green_lo, self.config.green_hi)
        )
        source = green
        if highlight:
            lo = (
                max(self.config.green_lo[0], self.config.player_highlight_hue_min),
                self.config.green_lo[1],
                max(self.config.green_lo[2], self.config.player_highlight_value_min),
            )
            source = cv2.inRange(hsv, lo, self.config.green_hi)
        rx, ry, rw, rh = region or (0, 0, w, h)
        search = source[ry : ry + rh, rx : rx + rw]
        merged = cv2.morphologyEx(
            search,
            cv2.MORPH_CLOSE,
            np.ones((max(1, round(2 * scale)), max(3, round(19 * scale))), np.uint8),
        )
        contours, _ = cv2.findContours(
            merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        names = []
        rejected = 0
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            x += rx
            y += ry
            minimum = self.config.player_name_min_width_px * scale
            if self.reference_width is not None:
                minimum = max(minimum, 0.60 * self.reference_width)
            if not minimum <= bw <= 250 * scale:
                continue
            # Efeitos verdes podem se ligar ao nome: isole sua faixa de glifos pela projeção horizontal.
            projection = np.count_nonzero(source[y : y + bh, x : x + bw], axis=1)
            active = np.flatnonzero(
                projection >= max(0.3 * bw, 0.40 * float(projection.max(initial=0)))
            )
            if not active.size:
                continue
            y0, y1 = max(0, int(active[0]) - round(2 * scale)), min(
                bh, int(active[-1]) + 1 + round(2 * scale)
            )
            y += y0
            bh = y1 - y0
            if not (
                minimum <= bw <= 250 * scale
                and max(3, 5 * scale) <= bh <= 30 * scale
                and 2.8 <= bw / bh <= 30
                and max(28 * scale, h * 0.13) <= y <= h * 0.88
            ):
                continue
            mask = source[y : y + bh, x : x + bw]
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            glyphs = sum(
                1
                for gx, gy, gw, gh, area in stats[1:]
                if gh >= max(2, round(4 * scale))
                and gw <= max(3, round(40 * scale))
                and area >= max(2, round(3 * scale * scale))
            )
            density = float(np.mean(mask > 0))
            if glyphs < 3 or not 0.06 <= density <= 0.75:
                rejected += 1
                continue
            # Assinatura comparável entre modo normal e fallback, sem alterar
            # a semântica de identidade quando só muda a iluminação.
            signature_mask = green[y : y + bh, x : x + bw]
            signature = (
                cv2.resize(
                    signature_mask, (96, 18), interpolation=cv2.INTER_AREA
                ).astype(float)
                / 255
            )
            similarity = 1.0
            if self.reference_name is not None:
                similarity = float(
                    np.sum(np.minimum(signature, self.reference_name))
                    / max(1e-6, np.sum(np.maximum(signature, self.reference_name)))
                )
                if similarity < self.config.player_name_similarity_min:
                    rejected += 1
                    continue
            names.append(((x, y, bw, bh), signature, similarity))
        self.diagnostics = {
            "name_candidates": len(names),
            "names_rejected_by_shape_or_reference": rejected,
            "fallback_without_name": False,
            "name_mask_mode": "highlight" if highlight else "regular",
        }
        return names

    def locate(self, hsv, timestamp_s=None, *, evidence=None):
        now = self.clock.resolve(timestamp_s)
        h, w = hsv.shape[:2]
        if self.resolution != (w, h):
            self.reset()
            self.resolution = (w, h)
        scale = w / self.config.reference_width
        if (
            self.last_seen is not None
            and now - self.last_seen > self.config.max_player_absence_s
        ):
            self.previous = None
            self.velocity[:] = 0

        def select(names):
            candidates = []
            for name, signature, similarity in names:
                for result in bar_observations(
                    hsv, name, self.config, evidence=evidence
                ):
                    score = result.pop("_score") + 0.18 * similarity
                    continuity = None
                    if self.previous is not None:
                        dt = max(0, now - self.last_seen)
                        predicted = np.array(self.previous) + self.velocity * min(
                            dt, 0.15
                        )
                        distance = float(
                            np.linalg.norm(np.array(result["center"]) - predicted)
                        )
                        gate = (
                            self.config.player_jump_base_px + 600 * min(dt, 0.5)
                        ) * scale
                        if distance > gate:
                            self.diagnostics["position_jump_rejected"] = (
                                self.diagnostics.get("position_jump_rejected", 0) + 1
                            )
                            continue
                        continuity = round(distance, 2)
                        score += 0.22 * (1 - distance / max(1, gate))
                    result["name_similarity"] = round(similarity, 3)
                    result["continuity_distance_px"] = continuity
                    candidates.append((score, result, signature))
            return candidates

        region = None
        if (
            self.previous is not None
            and self.last_global_at is not None
            and now - self.last_global_at < 0.35
        ):
            cx, cy = self.previous
            left, top = max(0, round(cx - 180 * scale)), max(0, round(cy - 100 * scale))
            right, bottom = min(w, round(cx + 180 * scale)), min(
                h, round(cy + 80 * scale)
            )
            region = (left, top, right - left, bottom - top)
        candidates = select(self._names(hsv, evidence, region))
        mode = "local" if region else "global"
        if not candidates and region:
            candidates = select(self._names(hsv, evidence))
            mode = "local_then_global"

        highlight_attempted = False
        highlight_used = False
        if not candidates and self.config.player_highlight_fallback:
            highlight_attempted = True
            candidates = select(self._names(hsv, evidence, region, highlight=True))
            if candidates:
                mode = "local_highlight" if region else "global_highlight"
                highlight_used = True
            elif region:
                candidates = select(self._names(hsv, evidence, highlight=True))
                if candidates:
                    mode = "local_then_global_highlight"
                    highlight_used = True

        if mode != "local":
            self.last_global_at = now
        self.diagnostics["search_mode"] = mode
        self.diagnostics["highlight_fallback_attempted"] = highlight_attempted
        self.diagnostics["highlight_fallback_used"] = highlight_used
        if candidates:
            _, result, signature = max(candidates, key=lambda v: v[0])
            if self.previous is not None and now > self.last_seen:
                dt = now - self.last_seen
                self.velocity = (
                    0.7 * self.velocity
                    + 0.3 * (np.array(result["center"]) - self.previous) / dt
                )
            self.previous = result["center"]
            self.last_seen = now
            self.missed = 0
            if self.reference_name is None:
                self.reference_name = signature
                self.reference_width = result["name_region"][2]
                normalized = np.packbits((signature >= 0.45).astype(np.uint8))
                digest = hashlib.blake2s(normalized.tobytes(), digest_size=10).hexdigest()
                self.name_visual_key = "visual:" + digest
            # A chave é uma assinatura visual persistente da faixa do nome,
            # NÃO a transcrição OCR do texto. A barra é observada neste quadro.
            result["name_identity_key"] = self.name_visual_key
            result["name_identity_source"] = "visual_template"
            result["name_text"] = None
            result["confidence"] = round(
                min(
                    0.95,
                    result["confidence"]
                    + 0.03 * (result["continuity_distance_px"] is not None),
                ),
                3,
            )
            return result
        self.missed += 1
        return None
