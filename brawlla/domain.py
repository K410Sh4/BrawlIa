"""Contratos de dados do núcleo. A serialização preserva campos JSON legados."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import math
import numpy as np


class GameState(StrEnum):
    UNKNOWN = "desconhecido"
    PREPARING = "pre_partida"
    COMBAT = "combate"
    DEFEAT = "derrota"
    RESULTS = "resultados"


class Timeline:
    """Relógio injetável: captura ao vivo ou tempo do vídeo; fallback nominal para API antiga."""

    def __init__(self, nominal_fps=15):
        self.step = 1 / nominal_fps
        self.last = None
        self.origin = None

    def resolve(self, timestamp=None, *, advance=True):
        value = (
            (0.0 if self.last is None else self.last + self.step)
            if timestamp is None
            else float(timestamp)
        )
        if (
            not math.isfinite(value)
            or value < 0
            or (self.last is not None and value < self.last - 1e-9)
        ):
            raise ValueError("Timestamp inválido ou fora de ordem")
        if advance:
            if self.origin is None:
                self.origin = value
            self.last = value
        return value


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    timestamp_s: float
    image: np.ndarray = field(repr=False, compare=False)
    capture_ms: float = 0.0
    metadata: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if (
            not math.isfinite(self.timestamp_s)
            or self.timestamp_s < 0
            or not math.isfinite(self.capture_ms)
            or self.capture_ms < 0
        ):
            raise ValueError("Metadados de quadro inválidos")
        if self.sequence < 1:
            raise ValueError("Sequência de quadro inválida")


@dataclass(frozen=True)
class Candidate:
    kind: str
    center: tuple[float, float]
    properties: dict = field(repr=False, compare=False)

    @classmethod
    def from_dict(cls, item):
        center = item["center"]
        kind = item["kind"]
        if (
            not isinstance(kind, str)
            or len(center) != 2
            or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in center)
        ):
            raise ValueError("Candidato inválido")
        return cls(kind, (float(center[0]), float(center[1])), dict(item))


@dataclass(frozen=True)
class StateStatus:
    name: str
    confidence: float
    raw: str
    held: bool = False
    uncertain: bool = False
    reason: str = ""
    evidence_age_s: float = 0.0
    pending: str | None = None
    pending_votes: int = 0
    changed: bool = False
    transition: dict | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TrackEvent:
    event: str
    timestamp_s: float
    reason: str
    track_id: int | None = None
    kind: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class AnalysisResult:
    resolution: tuple[int, int]
    game_state: dict
    observations: dict
    tracked_objects: list[dict]
    track_events: list[dict]
    timestamp_s: float
    session_elapsed_s: float
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self):
        from .config import VERSION, SCHEMA_VERSION

        return {
            **self.observations,
            "schema_version": SCHEMA_VERSION,
            "version": VERSION,
            "mode": "read_only",
            "resolution": list(self.resolution),
            "game_state": self.game_state,
            "tracked_objects": self.tracked_objects,
            "track_events": self.track_events,
            "analysis_timestamp_s": self.timestamp_s,
            "session_elapsed_s": round(self.session_elapsed_s, 6),
            "timings_ms": self.timings_ms,
        }
