"""Percepção V1: imagens regionais com coordenadas globais e contexto próprio.

Não reaproveita SessionAnalyzer em recortes: ele pressupõe quadro inteiro.
O detector regional é propositalmente conservador/heurístico, não um modelo treinado.
"""

from __future__ import annotations
from dataclasses import dataclass
import time
import cv2
import numpy as np

from brawlla.config import VisionConfig
from brawlla.state import StateDetector
from .geometry import Rect
from .context_engine import ContextMemory, Detection
from .hud_context import HUDContext, HUDAnchors, detect_controls
from .silhouette import extract_silhouette


@dataclass
class RegionImage:
    name: str
    rect: Rect
    image: np.ndarray

    def __post_init__(self):
        if (
            not isinstance(self.image, np.ndarray)
            or self.image.dtype != np.uint8
            or self.image.ndim != 3
            or self.image.shape[2] != 3
            or self.image.shape[:2] != (self.rect.h, self.rect.w)
        ):
            raise ValueError("Dimensão regional incompatível com coordenadas globais")


class RecognitionV1:
    def __init__(self, state_detector=None):
        self.state_detector = state_detector or StateDetector()
        self.hud = HUDContext()
        self.memory = ContextMemory()
        self.vision = VisionConfig()
        self.previous_state = "desconhecido"
        self.frames = 0
        self.full_frames = 0
        self.regional_frames = 0
        self.last_result = None

    @staticmethod
    def _hud_regions(regions: list[RegionImage], size: tuple[int, int]):
        width, height = size
        left = Rect(
            0, round(height * 0.60), round(width * 0.40), height - round(height * 0.60)
        )
        right = Rect(
            round(width * 0.60),
            round(height * 0.60),
            width - round(width * 0.60),
            height - round(height * 0.60),
        )
        anchors = HUDAnchors()
        for region in regions:
            for target in (left, right):
                overlap = region.rect.intersection_area(target)
                if overlap < target.area * 0.45:
                    continue
                x = max(region.rect.x, target.x)
                y = max(region.rect.y, target.y)
                w = min(region.rect.x + region.rect.w, target.x + target.w) - x
                h = min(region.rect.y + region.rect.h, target.y + target.h) - y
                crop = region.image[
                    y - region.rect.y : y - region.rect.y + h,
                    x - region.rect.x : x - region.rect.x + w,
                ]
                found = detect_controls(crop, (x, y), size)
                anchors = HUDAnchors(
                    attack=anchors.attack or found.attack,
                    movement=anchors.movement or found.movement,
                    gadget=anchors.gadget or found.gadget,
                )
        return anchors

    def _detect_bars(
        self, region: RegionImage, screen_size: tuple[int, int]
    ) -> list[Detection]:
        image = region.image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        scale = max(0.5, screen_size[0] / self.vision.reference_width)
        groups = (
            ("verde", [(self.vision.green_lo, self.vision.green_hi)]),
            ("amarela", [(self.vision.yellow_lo, self.vision.yellow_hi)]),
            (
                "vermelha",
                [
                    (self.vision.red_lo, self.vision.red_hi),
                    (self.vision.red_low_lo, self.vision.red_low_hi),
                ],
            ),
        )
        detections = []
        for color, ranges in groups:
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)),
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if (
                    w < 20 * scale
                    or w > 145 * scale
                    or h < 2
                    or h > 17 * scale
                    or w / h < 3.2
                ):
                    continue
                # Aparência de barra é apenas candidata; não infere HP exato.
                margin = max(2, round(5 * scale))
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2 = min(image.shape[1], x + w + margin)
                y2 = min(image.shape[0], y + h + margin)
                sample = hsv[y1:y2, x1:x2]
                if sample.size == 0:
                    continue
                darker = float(np.mean(sample[:, :, 2] < 100))
                confidence = min(0.78, 0.40 + 0.45 * darker)
                # Assinatura do corpo, não do texto da barra.
                body_w = round(max(w * 1.7, 38 * scale))
                body_h = round(max(w * 1.35, 50 * scale))
                bx1 = max(0, round(x + w / 2 - body_w / 2))
                by1 = min(image.shape[0], y + h)
                bx2 = min(image.shape[1], bx1 + body_w)
                by2 = min(image.shape[0], by1 + body_h)
                body = image[by1:by2, bx1:bx2]
                signature = extract_silhouette(body)
                detections.append(
                    Detection(
                        bbox=Rect(region.rect.x + x, region.rect.y + y, w, h),
                        kind="barra_vida_candidata",
                        color=color,
                        confidence=confidence,
                        signature=signature,
                        fill_fraction=None,
                    )
                )
        return detections

    @staticmethod
    def _deduplicate(detections: list[Detection]) -> list[Detection]:
        kept = []
        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            if any(
                det.bbox.intersection_area(other.bbox)
                >= min(det.bbox.area, other.bbox.area) * 0.45
                for other in kept
            ):
                continue
            kept.append(det)
        return kept

    def observe(
        self,
        regions: list[RegionImage],
        screen_size: tuple[int, int],
        now: float | None = None,
        *,
        full: bool = False,
    ) -> dict:
        now = time.perf_counter() if now is None else now
        if not regions:
            raise ValueError("Nenhuma região recebida")
        if full:
            if len(regions) != 1 or regions[0].rect != Rect(
                0, 0, screen_size[0], screen_size[1]
            ):
                raise ValueError("Varredura global requer quadro completo real")
        for region in regions:
            if region.rect.clipped(screen_size) != region.rect:
                raise ValueError("Região fora da janela")
        self.frames += 1
        if full:
            self.full_frames += 1
        else:
            self.regional_frames += 1
        anchors = self._hud_regions(regions, screen_size)
        visual = None
        if full:
            visual = self.state_detector.detect(regions[0].image, timestamp_s=now)
        state = self.hud.update(
            anchors,
            now,
            visual_state=(visual or {}).get("name"),
            visual_confidence=(visual or {}).get("confidence", 0.0),
        )
        event = None
        if self.previous_state != state["name"]:
            event = {
                "event": "state_change",
                "from": self.previous_state,
                "to": state["name"],
                "at": now,
            }
        if (
            state["name"] != "combate"
            and state["name"] in ("resultados", "derrota", "pre_partida")
            and self.previous_state == "combate"
        ):
            self.memory.reset("transicao_de_partida_confirmada")
        self.previous_state = state["name"]

        detections = []
        if state["name"] == "combate":
            for region in regions:
                # Evitar transformar os próprios botões do HUD em barras.
                if not full and region.name.startswith("hud_"):
                    continue
                detections.extend(self._detect_bars(region, screen_size))
        detections = self._deduplicate(detections)
        tracks = self.memory.update(
            detections, [r.rect for r in regions], now, screen_size
        )
        events = self.memory.events[:]
        self.memory.events.clear()
        if event:
            events.insert(0, event)
        result = {
            "recognition_version": "1.0",
            "frame_id": self.frames,
            "timestamp_s": now,
            "screen_resolution": list(screen_size),
            "capture_mode": "full" if full else "regional",
            "regions": [{"name": r.name, "xywh": r.rect.xywh()} for r in regions],
            "state": state,
            "visual_state": visual,
            "tracks": tracks,
            "events": events,
            "candidates_in_frame": len(detections),
            "limitations": [
                "Barras e silhuetas são hipóteses heurísticas.",
                "Não há leitura numérica exata de HP neste pipeline.",
                "Região não capturada nunca é declarada vazia.",
                "Não executa comandos no jogo.",
            ],
        }
        self.last_result = result
        return result
