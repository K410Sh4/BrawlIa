"""Brawlla v0.1 - visão computacional experimental (somente leitura).

Regras HSV calibradas em UMA captura de Combate do Brawl Stars. Não representam
um detector treinado e não garantem detecção correta em outros mapas/skins.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
from .player import PlayerLocator, white_number_support
from .config import VisionConfig, VERSION
from .health import other_bar_observations, FrameEvidence

import cv2
import numpy as np


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int
    area: int

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "center": list(self.center)}


def _range(
    hsv: np.ndarray, lo: tuple[int, int, int], hi: tuple[int, int, int]
) -> np.ndarray:
    return cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))


def _bars(
    hsv: np.ndarray,
    kind: str,
    diagnostics: dict | None = None,
    config=None,
    *,
    color_mask=None,
) -> list[Region]:
    """Reutiliza máscara HSV calculada no quadro quando disponível."""
    config = config or VisionConfig()
    if color_mask is not None:
        if color_mask.shape != hsv.shape[:2] or color_mask.dtype != np.uint8:
            raise ValueError("Máscara de cor inválida")
        mask = color_mask
    elif kind == "green":
        mask = _range(hsv, config.green_lo, config.green_hi)
    else:
        # Na amostra inicial, as barras de HP vermelhas são vermelho-magenta
        # (H>=169). A faixa H<11 confunde objetos alaranjados com barras.
        mask = _range(hsv, config.red_lo, config.red_hi)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = hsv.shape[:2]
    scale = w_img / config.legacy_reference_width
    result: list[Region] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if w < max(28, int(32 * scale)) or h < 3 or h > max(22, int(22 * scale)):
            continue
        if w / h < 2.7 or area / max(w * h, 1) < 0.32:
            continue
        if kind == "red":
            # Cabeçalho e rodapé podem conter ícones com as mesmas cores das barras.
            if (
                y < max(24, round(30 * w_img / config.reference_width))
                or y + h > h_img * 0.94
                or w > 170 * w_img / config.reference_width
            ):
                if diagnostics is not None:
                    diagnostics["outside_play_area"] = (
                        diagnostics.get("outside_play_area", 0) + 1
                    )
                continue
            # Tijolos rosados também geram retângulos vermelhos: exigir glifos claros próximos.
            if (
                white_number_support(
                    hsv,
                    x,
                    y,
                    max(w, round(65 * w_img / config.reference_width)),
                    config,
                )
                < config.number_support_min
            ):
                if diagnostics is not None:
                    diagnostics["missing_number_evidence"] = (
                        diagnostics.get("missing_number_evidence", 0) + 1
                    )
                continue
        # Desconsidera sobreposições praticamente iguais no mesmo frame.
        if any(
            abs(x + w / 2 - p.center[0]) < 12 * scale
            and abs(y + h / 2 - p.center[1]) < 12 * scale
            for p in result
        ):
            continue
        result.append(Region(x, y, w, h, int(area)))
    return sorted(result, key=lambda p: p.width * p.height, reverse=True)[:20]


def _terrain(
    hsv: np.ndarray, kind: str, config=None, *, include_contours=False
):
    config = config or VisionConfig()
    if kind == "bush":
        mask = _range(hsv, config.bush_lo, config.bush_hi)
        min_area = max(100, hsv.shape[0] * hsv.shape[1] * 0.00055)
    else:
        mask = _range(hsv, config.wall_lo, config.wall_hi)
        min_area = max(100, hsv.shape[0] * hsv.shape[1] * 0.00045)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = sorted(
        [c for c in contours if cv2.contourArea(c) >= min_area],
        key=cv2.contourArea,
        reverse=True,
    )[:30]
    reg = [Region(*cv2.boundingRect(c), int(cv2.contourArea(c))) for c in kept]
    if include_contours:
        # O observador de terreno já possui os contornos: não extraí-los
        # novamente após preencher uma máscara intermediária.
        return kept, reg
    region_mask = np.zeros_like(mask)
    cv2.drawContours(region_mask, kept, -1, 255, thickness=cv2.FILLED)
    return region_mask, reg


def detect(
    frame_bgr: np.ndarray,
    player_locator: PlayerLocator | None = None,
    timestamp_s=None,
    config=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        frame_bgr is None
        or not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.dtype != np.uint8
    ):
        raise ValueError("A imagem deve conter 3 canais de cor (BGR).")
    h, w = frame_bgr.shape[:2]
    if w < 320 or h < 200:
        raise ValueError("Imagem muito pequena; use captura com pelo menos 320x200 px.")
    config = config or VisionConfig()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    evidence = FrameEvidence(hsv, config)
    greens = _bars(hsv, "green", config=config, color_mask=evidence.color_masks[0][0])
    bar_filter = {}
    others = other_bar_observations(hsv, config, bar_filter, evidence=evidence)
    # Terreno em no máximo 640 px: contornos são reconvertidos às coordenadas originais.
    terrain_width = min(w, config.terrain_width)
    terrain_height = max(1, round(h * terrain_width / w))
    terrain_hsv = (
        cv2.resize(
            hsv, (terrain_width, terrain_height), interpolation=cv2.INTER_NEAREST
        )
        if w > config.terrain_width
        else hsv
    )
    scene = []
    regions = {}
    for kind in ("bush", "wall"):
        contours, _ = _terrain(terrain_hsv, kind, config, include_contours=True)
        scaled = [
            np.round(c * np.array([[[w / terrain_width, h / terrain_height]]])).astype(
                np.int32
            )
            for c in contours
        ]
        regions[kind] = [
            Region(*cv2.boundingRect(c), int(cv2.contourArea(c))) for c in scaled
        ]
        scene.append((kind, scaled))
    locator = player_locator or PlayerLocator(config)
    player = locator.locate(hsv, timestamp_s, evidence=evidence)
    if player:
        before = len(others)
        others = [
            r
            for r in others
            if not (
                abs(r["center"][1] - player["center"][1])
                < 12 * w / config.reference_width
                and player["x"] - 5 * w / config.reference_width
                <= r["center"][0]
                <= player["x"] + player["width"] + 5 * w / config.reference_width
            )
        ]
        bar_filter["overlaps_player"] = before - len(others)
    reds = [item for item in others if item["bar_color"] == "vermelha"]
    red_near = None
    if player and reds:
        red_near = min(
            reds,
            key=lambda item: np.hypot(
                item["center"][0] - player["center"][0],
                item["center"][1] - player["center"][1],
            ),
        )

    result: dict[str, Any] = {
        "version": VERSION,
        "mode": "read_only",
        "resolution": [w, h],
        "player_health_bar_candidate": player,
        "player_detection_diagnostics": locator.diagnostics,
        "visual_processing": evidence.diagnostics(),
        "other_green_health_bar_candidates": [
            p.to_dict()
            for p in greens
            if not player or abs(p.center[1] - player["center"][1]) > 10
        ],
        "red_bar_filter_rejections": bar_filter,
        "health_bar_filter_rejections": bar_filter,
        "red_health_bar_candidates": reds,
        "other_health_bar_candidates": others,
        "health_bar_candidates": ([player] if player else []) + others,
        "nearest_red_bar_candidate": red_near,
        "bush_regions": [p.to_dict() for p in regions["bush"]],
        "wall_regions": [p.to_dict() for p in regions["wall"]],
        "limitations": [
            "Regras de cores experimentais, calibradas em um único mapa e screenshot.",
            "Barras vermelhas podem indicar inimigos, caixas ou outros objetos.",
            "Sem OCR: não estima o número exato de pontos de vida.",
            "Não reconhece brawlers por classe e não executa comandos de jogo.",
        ],
    }
    return result, scene


def analyze(frame_bgr: np.ndarray, player_locator: PlayerLocator | None = None):
    """Compatibilidade da análise de cores; desenho delegado ao renderer."""
    from .rendering import render_terrain

    result, scene = detect(frame_bgr, player_locator)
    return render_terrain(frame_bgr, scene), result
