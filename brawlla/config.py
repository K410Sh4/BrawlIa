"""Configurações imutáveis, validadas e compartilhadas pelo núcleo."""

from dataclasses import dataclass, field, fields
import json
import math
import warnings
from pathlib import Path

VERSION = "1.1.0"
VERSION_SHORT = VERSION
SCHEMA_VERSION = 2
FPS_CHOICES = (5, 10, 15, 30, 60)


def positive(value, name, allow_zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        raise ValueError(
            f'{name}: número finito {"não negativo" if allow_zero else "positivo"} esperado'
        )


@dataclass(frozen=True)
class StateConfig:
    check_interval_s: float = 0.18
    enter_combat_s: float = 0.12
    confirm_menu_s: float = 0.40
    combat_grace_s: float = 2.5
    hud_threshold: float = 0.48
    defeat_threshold: float = 0.66
    results_threshold: float = 0.70
    menu_threshold: float = 0.78
    search_width: int = 640

    def __post_init__(self):
        for f in fields(self):
            positive(getattr(self, f.name), f.name)
        for name in (
            "hud_threshold",
            "defeat_threshold",
            "results_threshold",
            "menu_threshold",
        ):
            if getattr(self, name) > 1:
                raise ValueError(f"{name}: valor deve estar entre 0 e 1")
        if self.search_width < 320 or not isinstance(self.search_width, int):
            raise ValueError("search_width deve ser inteiro >= 320")


@dataclass(frozen=True)
class TrackingConfig:
    max_absence_s: float = 0.8
    name_reid_max_age_s: float = 6.0
    name_reid_min_similarity: float = 0.55
    name_reid_max_distance_px: float = 480.0
    base_distance_px: float = 35
    motion_budget_px_s: float = 600
    max_gate_px: float = 220
    prediction_horizon_s: float = 0.25
    velocity_tau_s: float = 0.12
    reference_width: int = 1280

    def __post_init__(self):
        for f in fields(self):
            positive(getattr(self, f.name), f.name)
        if self.max_gate_px < self.base_distance_px:
            raise ValueError("max_gate_px deve ser >= base_distance_px")
        if self.name_reid_max_age_s < self.max_absence_s:
            raise ValueError("name_reid_max_age_s deve ser >= max_absence_s")
        if self.name_reid_min_similarity > 1:
            raise ValueError("name_reid_min_similarity deve ser <= 1")


@dataclass(frozen=True)
class VisionConfig:
    reference_width: int = 1280
    legacy_reference_width: int = 1605
    terrain_width: int = 640
    number_support_min: float = 0.018
    min_container_fraction: float = 0.45
    min_short_fill_fraction: float = 0.30
    single_glyph_container_fraction: float = 0.60
    number_relaxed_support_factor: float = 0.40
    number_value_min: int = 180
    number_small_value_min: int = 165
    number_saturation_max: int = 85
    container_value_max: int = 85
    container_neutral_value_max: int = 140
    container_neutral_saturation_max: int = 80
    container_blue_lo: tuple = (100, 0, 0)
    container_blue_hi: tuple = (150, 199, 154)
    white_lo: tuple = (0, 0, 190)
    white_hi: tuple = (179, 80, 255)
    other_lo: tuple = (82, 95, 155)
    other_hi: tuple = (164, 255, 255)
    red_low_lo: tuple = (0, 95, 105)
    red_low_hi: tuple = (12, 255, 255)
    motion_enabled: bool = True
    player_name_min_width_px: float = 64
    player_name_similarity_min: float = 0.22
    player_jump_base_px: float = 45
    player_container_width_px: float = 76
    max_player_absence_s: float = 0.8
    player_highlight_fallback: bool = True
    player_highlight_hue_min: int = 64
    player_highlight_value_min: int = 200
    green_lo: tuple = (42, 110, 120)
    green_hi: tuple = (81, 255, 255)
    yellow_lo: tuple = (18, 90, 110)
    yellow_hi: tuple = (42, 255, 255)
    red_lo: tuple = (165, 95, 105)
    red_hi: tuple = (179, 255, 255)
    bush_lo: tuple = (85, 195, 70)
    bush_hi: tuple = (108, 255, 255)
    wall_lo: tuple = (160, 55, 195)
    wall_hi: tuple = (179, 150, 255)

    def __post_init__(self):
        if type(self.motion_enabled) is not bool:
            raise ValueError("motion_enabled deve ser booleano")
        if type(self.player_highlight_fallback) is not bool:
            raise ValueError("player_highlight_fallback deve ser booleano")
        for name, limit in (
            ("player_highlight_hue_min", 179),
            ("player_highlight_value_min", 255),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= limit:
                raise ValueError(f"{name}: inteiro fora do intervalo HSV")
        for name in ("reference_width", "legacy_reference_width", "terrain_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 320:
                raise ValueError(f"{name}: inteiro >= 320 esperado")
        for name in (
            "number_support_min",
            "min_container_fraction",
            "min_short_fill_fraction",
            "single_glyph_container_fraction",
            "number_relaxed_support_factor",
        ):
            positive(getattr(self, name), name)
            if getattr(self, name) > 1:
                raise ValueError(f"{name}: deve ser <= 1")
        for name in (
            "player_name_min_width_px",
            "player_name_similarity_min",
            "player_jump_base_px",
        ):
            positive(getattr(self, name), name)
        if self.player_name_similarity_min > 1:
            raise ValueError("Similaridade do nome deve ser <= 1")
        positive(self.player_container_width_px, "player_container_width_px")
        positive(self.max_player_absence_s, "max_player_absence_s")
        for name in (
            "number_value_min",
            "number_small_value_min",
            "number_saturation_max",
            "container_value_max",
            "container_neutral_value_max",
            "container_neutral_saturation_max",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError(f"{name}: inteiro de 0 a 255 esperado")
        for base in (
            "green",
            "yellow",
            "red",
            "red_low",
            "white",
            "other",
            "container_blue",
            "bush",
            "wall",
        ):
            lo, hi = getattr(self, base + "_lo"), getattr(self, base + "_hi")
            for v in (lo, hi):
                if (
                    not isinstance(v, (list, tuple))
                    or len(v) != 3
                    or any(
                        isinstance(n, bool) or not isinstance(n, int) or n < 0 or n > m
                        for n, m in zip(v, (179, 255, 255))
                    )
                ):
                    raise ValueError(f"{base}: faixa HSV inválida")
            if any(a > b for a, b in zip(lo, hi)):
                raise ValueError(f"{base}: faixa HSV invertida")
            object.__setattr__(self, base + "_lo", tuple(lo))
            object.__setattr__(self, base + "_hi", tuple(hi))


@dataclass(frozen=True)
class AnalyzerConfig:
    state: StateConfig = field(default_factory=StateConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)


def migrate_config(raw):
    """Remove apenas chaves que a V0.9 já ignorava; torna a migração explícita."""
    result = dict(raw)
    if "vision" in result:
        if not isinstance(result["vision"], dict):
            raise ValueError("vision deve ser um objeto")
        vision = dict(result["vision"])
        for name in ("container_dark_support_min", "container_dark_fraction"):
            if name in vision:
                vision.pop(name)
                warnings.warn(
                    f"vision.{name} foi removido: não tinha efeito na V0.9; use os parâmetros min_container_fraction/min_short_fill_fraction.",
                    FutureWarning,
                    stacklevel=2,
                )
        result["vision"] = vision
    return result


def load_config(path: Path | None = None) -> AnalyzerConfig:
    if path is None or not Path(path).exists():
        return AnalyzerConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or set(raw) - {"state", "tracking", "vision"}:
        raise ValueError("Seções de configuração inválidas")
    raw = migrate_config(raw)
    try:
        return AnalyzerConfig(
            **{
                name: cls(**raw.get(name, {}))
                for name, cls in (
                    ("state", StateConfig),
                    ("tracking", TrackingConfig),
                    ("vision", VisionConfig),
                )
            }
        )
    except TypeError as exc:
        raise ValueError(f"Campo de configuração inválido: {exc}") from exc


@dataclass(frozen=True)
class SessionOptions:
    auto_fps: bool = False
    capture_fps: int = 30
    analysis_fps: int = 15
    max_width: int = 1280
    record_samples: bool = False
    sample_fps: int = 2
    record_video: bool = False
    video_fps: int = 15
    roi: tuple[float, float, float, float] | None = None
    source_aspect: float = 0.0
    burst_fps: int = 10

    def __post_init__(self):
        if type(self.auto_fps) is not bool:
            raise ValueError("auto_fps deve ser booleano")
        if self.burst_fps not in (5, 10) or type(self.burst_fps) is not int:
            raise ValueError("Rajada deve ser 5 ou 10 FPS")
        if (
            not isinstance(self.source_aspect, (int, float))
            or not math.isfinite(self.source_aspect)
            or self.source_aspect < 0
        ):
            raise ValueError("Proporção da fonte inválida")
        if self.roi is not None:
            if len(self.roi) != 4 or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in self.roi
            ):
                raise ValueError("Recorte inválido")
            object.__setattr__(self, "roi", tuple(self.roi))
            x, y, w, h = self.roi
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001:
                raise ValueError("Recorte fora da imagem")
        for name in ("capture_fps", "analysis_fps", "video_fps"):
            if (
                type(getattr(self, name)) is not int
                or getattr(self, name) not in FPS_CHOICES
            ):
                raise ValueError(f"{name}: FPS inválido")
        if type(self.sample_fps) is not int or self.sample_fps not in (1, 2, 5, 10, 15):
            raise ValueError("sample_fps inválido")
        if (
            not isinstance(self.max_width, int)
            or isinstance(self.max_width, bool)
            or self.max_width < 320
        ):
            raise ValueError("max_width deve ser inteiro >= 320")
        if not isinstance(self.record_samples, bool) or not isinstance(
            self.record_video, bool
        ):
            raise ValueError("Opções de gravação devem ser booleanas")
