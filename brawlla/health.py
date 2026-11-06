"""Evidências locais de barras de vida: glifos, moldura e preenchimento.

Não faz OCR e não converte pixels em pontos de vida exatos.
"""

from __future__ import annotations

import cv2
import numpy as np


def number_evidence(hsv, x, y, width, scale=1, config=None, index=None):
    from .config import VisionConfig

    config = config or VisionConfig()
    if index is not None:
        return index.query(x, y, width)
    minimum = config.number_small_value_min if scale < 0.75 else config.number_value_min
    mask = cv2.inRange(hsv, (0, 0, minimum), (179, config.number_saturation_max, 255))
    return _number_from_mask(mask, x, y, width, scale)


class FrameEvidence:
    """Máscaras uma vez por quadro; cache local mantém os mesmos recortes de glifos."""

    def __init__(self, hsv, config):
        self.hsv = hsv
        self.config = config
        self.scale = hsv.shape[1] / config.reference_width
        self.color_masks = fill_masks(hsv, config)
        minimum = (
            config.number_small_value_min
            if self.scale < 0.75
            else config.number_value_min
        )
        self.number_mask = cv2.inRange(
            hsv, (0, 0, minimum), (179, config.number_saturation_max, 255)
        )
        self.numbers = self
        self.cache = {}
        self.queries = 0
        self.component_calls = 0

    def masks(self, top, bottom, left, right):
        return tuple(
            (mask[top:bottom, left:right], color) for mask, color in self.color_masks
        )

    def query(self, x, y, width):
        key = (round(x), round(y), round(width))
        self.queries += 1
        if key not in self.cache:
            self.component_calls += 1
            self.cache[key] = _number_from_mask(self.number_mask, *key, self.scale)
        return self.cache[key]

    def diagnostics(self):
        return {
            "number_queries": self.queries,
            "number_component_calls": self.component_calls,
            "number_cache_hits": self.queries - self.component_calls,
            "masks_built": 6,
        }


def _number_from_mask(mask, x, y, width, scale):
    h, w = mask.shape[:2]
    left, top = max(0, round(x)), max(0, round(y - 18 * scale))
    right, bottom = min(w, round(x + width)), min(h, round(y + 3 * scale))
    patch = mask[top:bottom, left:right]
    if not patch.size:
        return {"support": 0.0, "glyphs": 0, "baseline_gap_px": None}
    _, _, stats, _ = cv2.connectedComponentsWithStats(patch)
    glyphs = []
    for xx, yy, bw, bh, area in stats[1:]:
        if (
            max(2, round(3 * scale)) <= bh <= max(4, round(17 * scale))
            and max(1, round(scale)) <= bw <= max(3, round(45 * scale))
            and area >= max(2, round(3 * scale * scale))
            and bw / bh < 5
            and not (bw / bh > 1.8 and area / max(1, bw * bh) > 0.9)
            and abs(y - (top + yy + bh)) <= max(4, 8 * scale)
        ):
            glyphs.append((xx, yy, bw, bh, area))
    support = sum(g[-1] for g in glyphs) / patch.size
    gap = min((abs(y - (top + g[1] + g[3])) for g in glyphs), default=None)
    return {
        "support": round(float(support), 4),
        "glyphs": len(glyphs),
        "baseline_gap_px": round(float(gap), 2) if gap is not None else None,
    }


def has_number_support(evidence, config):
    # Glifos separados sustentam uma área clara menor após antialiasing/redimensionamento.
    minimum = config.number_support_min * (
        config.number_relaxed_support_factor if evidence["glyphs"] >= 2 else 1.0
    )
    return evidence["glyphs"] >= 1 and evidence["support"] >= minimum


def background_mask(hsv, config=None):
    """Inclui o recipiente azul-escuro visível quando o preenchimento é mínimo."""
    from .config import VisionConfig

    config = config or VisionConfig()
    _, ss, vv = cv2.split(hsv)
    return (
        (
            (vv < config.container_value_max)
            | (
                (vv < config.container_neutral_value_max)
                & (ss < config.container_neutral_saturation_max)
            )
        )
        | (cv2.inRange(hsv, config.container_blue_lo, config.container_blue_hi) > 0)
    ).astype(np.uint8) * 255


def fill_masks(hsv, config):
    green = cv2.inRange(hsv, config.green_lo, config.green_hi)
    yellow = cv2.inRange(hsv, config.yellow_lo, config.yellow_hi)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, config.red_lo, config.red_hi),
        cv2.inRange(hsv, config.red_low_lo, config.red_low_hi),
    )
    white = cv2.inRange(hsv, config.white_lo, config.white_hi)
    other = cv2.inRange(hsv, config.other_lo, config.other_hi)
    return (
        (green, "verde"),
        (yellow, "amarela"),
        (red, "vermelha"),
        (white, "branca"),
        (other, "outra"),
    )


def bar_observations(hsv, name, config, *, evidence=None):
    """Observa pixels atuais abaixo de um nome; nunca retorna previsão sem imagem."""
    nx, ny, nw, nh = name
    h, w = hsv.shape[:2]
    scale = w / config.reference_width
    expected = max(12, round(config.player_container_width_px * scale))
    center = nx + nw / 2
    left = max(0, round(center - expected * 0.85))
    right = min(w, round(center + expected * 0.85))
    top = max(0, round(ny + nh))
    bottom = min(h, round(top + 50 * scale))
    if bottom - top < 3 or right - left < 8:
        return []
    patch = hsv[top:bottom, left:right]
    seeds = []
    for mask, color in (
        evidence.masks(top, bottom, left, right)
        if evidence
        else fill_masks(patch, config)
    ):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if (
                max(1, round(scale)) <= bw <= 115 * scale
                and max(2, round(3 * scale))
                <= bh
                <= (26 if bw >= expected * 0.7 else 21) * scale
                and x > 0
                and x + bw < right - left
                and (
                    bw <= max(2, round(2 * scale))
                    or cv2.contourArea(contour) / max(1, bw * bh) >= 0.30
                )
            ):
                seeds.append((x + left, y + top, bw, bh, color, False))
    dark = background_mask(patch, config)
    # A abertura horizontal retira glifos/linhas verticais, preservando o recipiente.
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        np.ones((max(1, round(2 * scale)), max(3, round(18 * scale))), np.uint8),
    )
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_CLOSE, np.ones((1, max(3, round(10 * scale))), np.uint8)
    )
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if (
            0.35 * expected <= bw <= 1.5 * expected
            and max(2, round(3 * scale)) <= bh <= 21 * scale
        ):
            seeds.append((x + left, y + top, bw, bh, "indeterminada", True))
    observations = []
    seen = set()
    for sx, sy, bw, bh, color, is_dark in seeds:
        # O centro segue o recipiente/nome, não o centro de um preenchimento que encolhe.
        full_width = (
            max(expected, bw) if bw >= expected * 0.8 and not is_dark else expected
        )
        x = max(0, min(w - full_width, round(center - full_width / 2)))
        if not x - 5 * scale <= sx <= x + full_width * 0.55 and bw < expected * 0.7:
            continue  # Preenchimento curto precisa nascer junto da borda esquerda.
        if sx + bw < x or sx > x + full_width:
            continue
        key = (x, sy, full_width, bh, color)
        if key in seen:
            continue
        seen.add(key)
        number = number_evidence(
            hsv,
            x,
            sy,
            full_width,
            scale,
            config,
            evidence.numbers if evidence else None,
        )
        inner = hsv[sy : min(h, sy + bh), x : min(w, x + full_width)]
        if not inner.size:
            continue
        dark_fraction = float(np.mean(background_mask(inner, config) > 0))
        single_glyph_with_container = (
            number["glyphs"] == 1
            and number["support"]
            >= config.number_support_min * config.number_relaxed_support_factor
            and dark_fraction >= config.single_glyph_container_fraction
        )
        if not has_number_support(number, config) and not single_glyph_with_container:
            continue
        if is_dark and dark_fraction < config.min_container_fraction:
            continue
        if (
            not is_dark
            and bw < expected * 0.3
            and dark_fraction < config.min_short_fill_fraction
        ):
            continue
        if color == "branca" and (bw < expected * 0.40 or bw / max(1, bh) < 2.5):
            continue
        # Rejeita sementes distantes da linha sob o nome; munição não tem os glifos na sua linha.
        gap = (sy - (ny + nh)) / max(scale, 1e-6)
        if not 1 <= gap <= 30:
            continue
        score = (
            0.46
            + min(0.18, number["support"] * 1.8)
            + min(0.10, number["glyphs"] * 0.025)
            + (0.08 if not is_dark else 0.02)
            + min(0.08, dark_fraction * 0.08)
            + (
                min(0.07, bw * bh / max(1, expected * 10 * scale) * 0.07)
                if not is_dark
                else 0
            )
        )
        observations.append(
            {
                "x": x,
                "y": sy,
                "width": full_width,
                "height": bh,
                "area": int(bw * bh),
                "center": [x + full_width // 2, sy + bh // 2],
                "filled_width_px": None if is_dark else bw,
                "fill_fraction_proxy": (
                    None if is_dark else round(min(1.0, bw / full_width), 3)
                ),
                "health_status": (
                    "fill_not_measured"
                    if is_dark
                    else (
                        "low_fill_candidate"
                        if bw / full_width <= 0.25
                        else "fill_visible"
                    )
                ),
                "bar_color": color,
                "container_width_estimated": True,
                "container_dark_fraction": round(dark_fraction, 3),
                "number_support": number["support"],
                "number_evidence": number,
                "health_evidence": {
                    "name": True,
                    "numbers": True,
                    "container": bool(dark_fraction >= config.min_short_fill_fraction),
                    "colored_fill": not is_dark,
                    "current_frame_observed": True,
                },
                "confidence": round(min(0.9, score), 3),
                "player_evidence": (
                    "nome_verde_glifos_e_moldura" if is_dark else "nome_verde_e_barra"
                ),
                "name_region": list(name),
                "_score": score,
            }
        )
    return observations


def other_bar_observations(hsv, config, diagnostics=None, *, evidence=None):
    """Barras de outros alvos; cor não determina identidade nem tipo de objeto."""
    h, w = hsv.shape[:2]
    scale = w / config.reference_width
    expected = round(config.player_container_width_px * scale)
    result = []
    # Verde é reservado às propostas separadas de aliados/jogador. As outras cores exigem glifos.
    for mask, color in (evidence.color_masks if evidence else fill_masks(hsv, config))[
        1:
    ]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        eligible = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if (
                max(1, round(scale)) <= bw <= 150 * scale
                and max(2, round(3 * scale)) <= bh <= 20 * scale
                and y >= max(24, 30 * scale)
                and y + bh < h * 0.94
                and (
                    bw <= max(2, round(2 * scale))
                    or cv2.contourArea(contour) / max(1, bw * bh) >= 0.38
                )
            ):
                eligible.append((x, y, bw, bh))
        for x, y, bw, bh in sorted(eligible, key=lambda r: r[2] * r[3], reverse=True)[
            :80
        ]:
            small = bw < 28 * scale
            if not small and bw / bh < 2.7:
                continue
            full_width = min(w - x, expected if small else bw)
            numbers = number_evidence(
                hsv,
                x,
                y,
                full_width,
                scale,
                config,
                evidence.numbers if evidence else None,
            )
            if not has_number_support(numbers, config):
                if diagnostics is not None:
                    diagnostics["missing_number_evidence"] = (
                        diagnostics.get("missing_number_evidence", 0) + 1
                    )
                continue
            patch = hsv[y : y + bh, x : x + full_width]
            dark_fraction = float(np.mean(background_mask(patch, config) > 0))
            if small and dark_fraction < config.min_container_fraction:
                if diagnostics is not None:
                    diagnostics["missing_container_evidence"] = (
                        diagnostics.get("missing_container_evidence", 0) + 1
                    )
                continue
            if color == "branca" and small:
                continue
            if any(
                abs(x + full_width / 2 - r["center"][0]) < 18 * scale
                and abs(y + bh / 2 - r["center"][1]) < 12 * scale
                for r in result
            ):
                continue
            result.append(
                {
                    "x": x,
                    "y": y,
                    "width": full_width,
                    "height": bh,
                    "area": bw * bh,
                    "center": [x + full_width // 2, y + bh // 2],
                    "bar_color": color,
                    "filled_width_px": bw,
                    "container_width_estimated": small,
                    "number_support": numbers["support"],
                    "number_evidence": numbers,
                    "health_evidence": {
                        "numbers": True,
                        "container": dark_fraction >= config.min_short_fill_fraction,
                        "colored_fill": True,
                        "current_frame_observed": True,
                    },
                    "confidence": 0.65 if not small else 0.55,
                }
            )
    return result[:30]
