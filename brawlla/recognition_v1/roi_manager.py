"""Planejador de busca regional com recuperação global periódica."""
from __future__ import annotations
from dataclasses import dataclass
from .geometry import Rect, CaptureRegion


@dataclass
class CapturePlan:
    full: bool
    regions: list[CaptureRegion]
    reason: str
    coverage: float

    def to_dict(self) -> dict:
        return {"full": self.full, "regions": [r.to_dict() for r in self.regions],
                "reason": self.reason, "coverage": round(self.coverage, 4)}


class ROIManager:
    def __init__(self, full_interval_s: float = 0.75,
                 max_regions: int = 8, max_coverage: float = 0.55):
        if full_interval_s <= 0 or max_regions < 1 or not 0 < max_coverage <= 1:
            raise ValueError("Política de ROI inválida")
        self.full_interval_s = full_interval_s
        self.max_regions = max_regions
        self.max_coverage = max_coverage
        self.last_full: float | None = None
        self.force_reason: str | None = None
        self.full_count = 0
        self.regional_count = 0

    def request_full_scan(self, reason: str):
        self.force_reason = reason

    def note_full(self, now: float):
        self.last_full = now
        self.force_reason = None
        self.full_count += 1

    @staticmethod
    def _merge(regions: list[CaptureRegion]) -> list[CaptureRegion]:
        merged = []
        for region in sorted(regions, key=lambda r: r.priority, reverse=True):
            rect = region.rect
            reason = region.reason
            found = True
            while found:
                found = False
                for i, old in enumerate(merged):
                    if rect.intersection_area(old.rect):
                        rect = rect.merged(old.rect)
                        reason += "+" + old.reason
                        merged.pop(i)
                        found = True
                        break
            merged.append(CaptureRegion(region.name, rect, reason, region.priority))
        return merged

    def plan(self, now: float, screen_size: tuple[int, int],
             state: str, tracks: list[dict]) -> CapturePlan:
        width, height = screen_size
        if width < 320 or height < 200:
            raise ValueError("Dimensão inválida")
        forced = self.force_reason
        if forced or self.last_full is None or now - self.last_full >= self.full_interval_s:
            return CapturePlan(True, [], forced or "varredura_periodica", 1.0)
        if state != "combate":
            return CapturePlan(True, [], "estado_fora_de_combate", 1.0)
        hud = [
            CaptureRegion("hud_left", Rect(0, round(height * 0.60),
                          round(width * 0.40), height - round(height * 0.60)),
                          "ancora_movimento", 2),
            CaptureRegion("hud_right", Rect(round(width * 0.60), round(height * 0.60),
                          width - round(width * 0.60), height - round(height * 0.60)),
                          "ancora_ataque", 2),
        ]
        regions = list(hud)
        for track in tracks:
            if track["status"] == "perdido":
                continue
            x, y, w, h = track["xywh"]
            bbox = Rect(int(x), int(y), int(w), int(h))
            dt = max(0, now - track["last_seen"])
            vx, vy = track.get("velocity", [0.0, 0.0])
            predicted = bbox.translated(vx * min(dt, 0.35), vy * min(dt, 0.35), screen_size)
            if predicted is None:
                continue
            margin = 0.45 if track["status"] != "observado" else 0.30
            rect = predicted.expand(margin, screen_size)
            if rect is not None:
                # detector deve receber corpo+barra com espaço para contexto.
                regions.append(CaptureRegion("target_" + str(track["track_id"]),
                                             rect, "rastreio", 3))
        merged = self._merge(regions)
        coverage = sum(r.rect.area for r in merged) / (width * height)
        if len(merged) > self.max_regions or coverage >= self.max_coverage:
            return CapturePlan(True, [], "regioes_excessivas", 1.0)
        self.regional_count += 1
        return CapturePlan(False, merged, "captura_estrategica", coverage)
