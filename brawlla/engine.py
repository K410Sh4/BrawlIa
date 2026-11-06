"""Orquestra detectores, política de sessão e tracker; desenho fica em rendering."""

import time
import numpy as np
from .config import AnalyzerConfig
from .domain import AnalysisResult, Timeline
from .state import StateDetector
from .player import PlayerLocator
from .motion import MotionEvidence
from .objects import ObjectClassifier
from .tracking import TargetTracker
from .vision import detect
from .rendering import render_analysis
from .diagnostics import target_status

LIMITATIONS = [
    "Classificação heurística por exemplos visuais; não mede acurácia.",
    "IDs usam geometria e, quando disponível, assinatura visual do nome verde do jogador; isso não é OCR nem identidade pessoal confirmada.",
    "Confiança é indicador interno; candidatos não são identidades confirmadas.",
    "Detecção de telas depende de fonte, idioma e escala.",
    "Durante tolerância de HUD, o combate é marcado como incerto; não há extrapolação de observações.",
    "Não executa comandos no jogo.",
]


class SessionAnalyzer:
    def __init__(self, config=None):
        self.config = config or AnalyzerConfig()
        self.clock = Timeline()
        self.states = StateDetector(self.config.state)
        self.player = PlayerLocator(self.config.vision)
        self.motion = MotionEvidence()
        self.objects = ObjectClassifier(self.config.vision)
        self.tracker = TargetTracker(config=self.config.tracking)
        self.previous_state = "desconhecido"
        self.previous_resolution = None

    def observe(self, frame_bgr, timestamp_s=None):
        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.dtype != np.uint8
        ):
            raise ValueError("Frame BGR inválido")
        h, w = frame_bgr.shape[:2]
        if w < 320 or h < 200:
            raise ValueError("Frame deve ter pelo menos 320×200 px")
        now = self.clock.resolve(timestamp_s)
        started = time.perf_counter()
        state = self.states.detect(frame_bgr, timestamp_s=now)
        timings = {"state": (time.perf_counter() - started) * 1000}
        if self.previous_resolution not in (None, (w, h)):
            self.tracker.reset("resolucao_alterada", now)
            self.player.reset()
            self.motion.reset()
        self.previous_resolution = (w, h)
        if state["name"] != self.previous_state and state["name"] in (
            "pre_partida",
            "derrota",
            "resultados",
        ):
            self.tracker.reset(state["name"] + "_confirmado", now)
            self.player.reset()
            self.motion.reset()
        self.previous_state = state["name"]
        scene = []
        camera_start = time.perf_counter()
        camera = (
            self.motion.prepare(frame_bgr, now)
            if state["name"] == "combate" and self.config.vision.motion_enabled
            else {"valid": False, "reason": "disabled_or_outside_combat"}
        )
        timings["camera_motion"] = (time.perf_counter() - camera_start) * 1000
        if state["name"] == "combate":
            stage = time.perf_counter()
            data, scene = detect(frame_bgr, self.player, now, self.config.vision)
            timings["colors"] = (time.perf_counter() - stage) * 1000
            stage = time.perf_counter()
            objects = [
                {**item, **self.objects.classify(frame_bgr, item)}
                for item in data["other_health_bar_candidates"]
            ]
            player = data["player_health_bar_candidate"]
            if player:
                objects.insert(
                    0, {**player, "kind": "jogador_candidato", "box_evidence": 0.0}
                )
            timings["classification"] = (time.perf_counter() - stage) * 1000
        else:
            data = {
                "player_health_bar_candidate": None,
                "other_green_health_bar_candidates": [],
                "red_health_bar_candidates": [],
                "other_health_bar_candidates": [],
                "health_bar_candidates": [],
                "player_detection_diagnostics": {},
                "nearest_red_bar_candidate": None,
                "bush_regions": [],
                "wall_regions": [],
                "red_bar_filter_rejections": {},
            }
            objects = []
        stage = time.perf_counter()
        tracked = self.tracker.update(objects, w, now, camera_motion=camera)
        timings["association"] = (time.perf_counter() - stage) * 1000
        timings["tracking"] = timings[
            "association"
        ]  # Compatibilidade, agora sem fluxo óptico incluído.
        stage = time.perf_counter()
        if state["name"] == "combate" and self.config.vision.motion_enabled:
            self.motion.update(frame_bgr, tracked, now)
        timings["body_motion"] = (time.perf_counter() - stage) * 1000
        data["camera_motion"] = camera
        for item in tracked:
            if item["kind"] == "jogador_candidato":
                motion = item.get("motion_evidence", {})
                item["health_evidence"]["body_comotion"] = motion.get(
                    "body_bar_consistent", False
                )
                if motion.get("classification") == "moving_with_body":
                    item["confidence"] = min(0.95, item["confidence"] + 0.02)
                data["player_health_bar_candidate"]["motion_evidence"] = motion
        data["objects_count_by_kind"] = {
            kind: sum(o["kind"] == kind for o in tracked)
            for kind in (
                "jogador_candidato",
                "caixa_provavel",
                "personagem_possivel",
                "indefinido",
            )
        }
        data["target_analysis"] = target_status(state)
        data["limitations"] = LIMITATIONS.copy()
        timings["core"] = (time.perf_counter() - started) * 1000
        events = self.tracker.drain_events()
        for event in events:
            event["session_elapsed_s"] = round(
                event["timestamp_s"] - self.clock.origin, 6
            )
        result = AnalysisResult(
            (w, h),
            state,
            data,
            tracked,
            events,
            now,
            now - self.clock.origin,
            {key: round(value, 3) for key, value in timings.items()},
        )
        return result, scene

    def analyze(self, frame_bgr, timestamp_s=None, *, render=True):
        started = time.perf_counter()
        result, scene = self.observe(frame_bgr, timestamp_s)
        data = result.to_dict()
        stage = time.perf_counter()
        image = render_analysis(frame_bgr, data, scene) if render else frame_bgr.copy()
        data["timings_ms"]["overlay"] = round((time.perf_counter() - stage) * 1000, 3)
        data["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000, 3)
        return image, data


def analyze(frame_bgr):
    """Imagem isolada. Reutilize SessionAnalyzer para preservar histórico."""
    return SessionAnalyzer().analyze(frame_bgr)
