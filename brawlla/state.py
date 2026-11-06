"""Classificação conservadora de telas, baseada em exemplos visuais locais.

Os exemplos de texto foram extraídos de uma gravação fornecida pelo usuário.
Não usa OCR nem dados privados; a mesma aparência de fonte pode mudar com
idioma, resolução ou atualizações do jogo.
"""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .config import StateConfig
from .domain import GameState, StateStatus, Timeline

ASSETS = Path(__file__).resolve().parent / "assets"


def _load_template(name: str) -> np.ndarray:
    im = cv2.imread(str(ASSETS / name), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Template de interface ausente: {name}")
    return im


class StateStabilizer:
    """Política temporal: ausência do HUD é incerteza, nunca evidência de menu."""

    def __init__(self, config=None):
        self.config = config or StateConfig()
        self.clock = Timeline()
        self.state = GameState.UNKNOWN.value
        self.confidence = 0.0
        self.pending = None
        self.pending_since = None
        self.votes = 0
        self.last_combat = None
        self.initialized = False

    def update(
        self, raw: str, confidence: float, timestamp_s=None, *, evidence_checked=True
    ) -> dict:
        raw = GameState(raw).value
        now = self.clock.resolve(timestamp_s)
        previous = self.state
        reason = "evidencia_confirmada"
        if raw == "combate" and evidence_checked:
            self.last_combat = now
        if raw in ("derrota", "resultados"):
            self.state, self.confidence = raw, confidence
            self.pending, self.pending_since, self.votes = None, None, 0
            reason = "texto_terminal_detectado"
        elif raw == self.state:
            if raw == "desconhecido":
                reason = "sem_evidencia_visual"
            self.confidence = confidence
            self.pending, self.pending_since, self.votes = None, None, 0
        elif raw == "desconhecido":
            self.pending, self.pending_since, self.votes = None, None, 0
            if (
                self.state == "combate"
                and self.last_combat is not None
                and now - self.last_combat <= self.config.combat_grace_s + 1e-9
            ):
                reason = "hud_temporariamente_ausente"
            else:
                self.state, self.confidence = "desconhecido", confidence
                reason = (
                    "limite_sem_evidencia_excedido"
                    if previous == "combate"
                    else "sem_evidencia_visual"
                )
        else:
            if self.pending != raw:
                self.pending, self.pending_since, self.votes = raw, now, 1
            else:
                self.votes += 1
            delay = (
                self.config.confirm_menu_s
                if raw == "pre_partida"
                else self.config.enter_combat_s
            )
            if previous == "desconhecido" and not self.initialized:
                delay = 0  # Uma imagem isolada pode ser analisada sem histórico.
            if now - self.pending_since + 1e-9 >= delay:
                self.state, self.confidence = raw, confidence
                self.pending, self.pending_since, self.votes = None, None, 0
                reason = "menu_confirmado" if raw == "pre_partida" else "hud_confirmado"
            else:
                reason = (
                    "confirmando_menu"
                    if raw == "pre_partida"
                    else "confirmando_combate"
                )
        self.initialized = True
        changed = previous != self.state
        age = (
            max(0.0, now - self.last_combat)
            if self.last_combat is not None and self.state == "combate"
            else 0.0
        )
        transition = (
            {"from": previous, "to": self.state, "reason": reason, "timestamp_s": now}
            if changed
            else None
        )
        return StateStatus(
            self.state,
            round(self.confidence, 3),
            raw,
            raw != self.state,
            raw == "desconhecido",
            reason,
            round(age, 3),
            self.pending,
            self.votes,
            changed,
            transition,
        ).to_dict()


class StateDetector:
    def __init__(self, config=None) -> None:
        self.config = config or StateConfig()
        self.clock = Timeline()
        self._checked_at = None
        self._last_scores = None
        self.hud = _load_template("hud_combate.png")
        self.defeat = _load_template("derrota.png")
        self.results = _load_template("resultados.png")
        self.menu = _load_template("menu_jogar.png")
        self._templates = {
            name: self._compile(template)
            for name, template in (
                ("hud", self.hud),
                ("defeat", self.defeat),
                ("results", self.results),
                ("menu", self.menu),
            )
        }
        self.stabilizer = StateStabilizer(self.config)
        self._calls = 0
        self._last_result = None

    def _compile(self, template):
        ref = cv2.inRange(
            cv2.cvtColor(template, cv2.COLOR_BGR2HSV), (0, 0, 140), (179, 115, 255)
        )
        return [
            cv2.resize(
                ref,
                (
                    max(
                        2,
                        round(ref.shape[1] * factor * self.config.search_width / 1280),
                    ),
                    max(
                        2,
                        round(ref.shape[0] * factor * self.config.search_width / 1280),
                    ),
                ),
                interpolation=cv2.INTER_AREA,
            )
            for factor in (0.85, 1.0, 1.17, 1.35, 1.55)
        ]

    def _score(self, hsv, key, rect):
        x1, y1, x2, y2 = rect
        patch = hsv[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if not patch.size:
            return 0.0
        src = cv2.inRange(patch, (0, 0, 140), (179, 115, 255))
        best = 0.0
        for ref in self._templates[key]:
            if ref.shape[0] > src.shape[0] or ref.shape[1] > src.shape[1]:
                continue
            score = cv2.minMaxLoc(cv2.matchTemplate(src, ref, cv2.TM_CCOEFF_NORMED))[1]
            best = max(best, float(score))
        return round(best, 3)

    def detect(self, frame: np.ndarray, update: bool = True, timestamp_s=None) -> dict:
        now = self.clock.resolve(timestamp_s, advance=update)
        if update:
            self._calls += 1
        cached = (
            update
            and self._checked_at is not None
            and now - self._checked_at < self.config.check_interval_s - 1e-9
        )
        if cached:
            scores = self._last_scores.copy()
        else:
            h, w = frame.shape[:2]
            width = self.config.search_width
            base = cv2.resize(
                frame,
                (width, max(1, round(h * width / w))),
                interpolation=cv2.INTER_AREA,
            )
            hh, ww = base.shape[:2]
            hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
            scores = {
                "hud_score": self._score(
                    hsv, "hud", (0, 0, int(ww * 0.65), min(hh, round(63 * width / 640)))
                ),
                "defeat_score": self._score(
                    hsv,
                    "defeat",
                    (int(ww * 0.12), int(hh * 0.10), int(ww * 0.92), int(hh * 0.60)),
                ),
                "results_score": self._score(
                    hsv, "results", (0, 0, int(ww * 0.65), max(1, int(hh * 0.30)))
                ),
                "menu_score": self._score(
                    hsv, "menu", (int(ww * 0.60), int(hh * 0.68), ww, hh)
                ),
            }
            if update:
                self._checked_at, self._last_scores = now, scores.copy()
        if scores["results_score"] >= self.config.results_threshold:
            raw, confidence = "resultados", scores["results_score"]
        elif scores["defeat_score"] >= self.config.defeat_threshold:
            raw, confidence = "derrota", scores["defeat_score"]
        elif scores["menu_score"] >= self.config.menu_threshold:
            raw, confidence = "pre_partida", scores["menu_score"]
        elif scores["hud_score"] >= self.config.hud_threshold:
            raw, confidence = "combate", scores["hud_score"]
        else:
            raw, confidence = "desconhecido", max(scores.values())
        stable = (
            self.stabilizer.update(raw, confidence, now, evidence_checked=not cached)
            if update
            else StateStatus(
                raw,
                confidence,
                raw,
                uncertain=raw == "desconhecido",
                reason="imagem_isolada",
            ).to_dict()
        )
        result = {
            **stable,
            **scores,
            "raw_confidence": round(confidence, 3),
            "method": "templates_visuais_heuristicos",
            "cached": cached,
        }
        if update:
            self._last_result = result
        return result
