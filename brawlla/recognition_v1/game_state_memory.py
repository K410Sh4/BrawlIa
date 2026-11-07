"""Persistência de estados por evidências explícitas.

Não observar != observar ausência. Um resultado confirmado continua sendo
resultado durante as aquisições regionais. Mudanças de tela exigem evidência
nova, com dupla confirmação ao sair de estados terminais.
"""

from __future__ import annotations

from typing import Any

CONFIRMED_STATES = frozenset(("combate", "derrota", "resultados", "pre_partida"))
TERMINAL_STATES = frozenset(("derrota", "resultados"))


class GameStateMemory:
    """Máquina de estados temporal, independente do tamanho de cada ROI."""

    def __init__(self, grace_s: float = 2.5, transition_votes: int = 2):
        if grace_s <= 0 or transition_votes < 1:
            raise ValueError("Parâmetros de contexto inválidos")
        self.grace_s = grace_s
        self.transition_votes = transition_votes
        self.state = "desconhecido"
        self.last_combat: float | None = None
        self.last_confirmed: float | None = None
        self.anchor_streak = 0
        self.confidence = 0.0
        self._confirmed_confidence = 0.0
        self._pending: str | None = None
        self._pending_votes = 0

    def _confirm(self, state: str, now: float, confidence: float):
        self.state = state
        self.confidence = max(0.5, min(1.0, float(confidence)))
        self.last_confirmed = now
        self._confirmed_confidence = self.confidence
        if state == "combate":
            self.last_combat = now
        self._pending = None
        self._pending_votes = 0

    def confirm_transition(self, state: str, now: float, confidence: float) -> bool:
        """Confirma uma mudança observada (não é acionada por ROI não checada)."""
        if state not in CONFIRMED_STATES:
            return False
        if state == self.state:
            self._confirm(state, now, confidence)
            return True
        # A sequência derrota -> resultados pode acontecer após uma única
        # observação global já estabilizada pelo StateDetector existente.
        if self.state not in TERMINAL_STATES or (
            self.state == "derrota" and state == "resultados"
        ):
            self._confirm(state, now, confidence)
            return True
        if self._pending != state:
            self._pending, self._pending_votes = state, 1
        else:
            self._pending_votes += 1
        if self._pending_votes >= self.transition_votes:
            self._confirm(state, now, confidence)
            return True
        return False

    def preserve_last_state(self, now: float) -> bool:
        if self.state in TERMINAL_STATES:
            return True
        return (
            self.state == "combate"
            and self.last_combat is not None
            and now - self.last_combat <= self.grace_s
        )

    def evaluate_state_confidence(self, now: float) -> float:
        """Confiança operacional relativa, não probabilidade calibrada."""
        if self.state in TERMINAL_STATES:
            if self.last_confirmed is None:
                elapsed_s = 0.0
            else:
                elapsed_s = max(0.0, now - self.last_confirmed)
            return max(0.50, self._confirmed_confidence * (0.93**elapsed_s))
        if self.state == "combate" and self.last_combat is not None:
            ratio = min(1.0, max(0.0, (now - self.last_combat) / self.grace_s))
            return max(0.20, self.confidence * (1.0 - 0.60 * ratio))
        return 0.0

    def observe_state(
        self,
        anchors: Any | None,
        now: float,
        visual_state: str | None = None,
        visual_confidence: float = 0.0,
    ) -> dict:
        """anchors=None: NOT_CHECKED; objeto falso: ABSENT; verdadeiro: PRESENT."""
        if anchors is None:
            anchor_observation = "NOT_CHECKED"
        elif bool(getattr(anchors, "combat_candidate", False)):
            anchor_observation = "PRESENT"
            self.anchor_streak += 1
        else:
            anchor_observation = "ABSENT"
            self.anchor_streak = 0

        visual_observation = (
            "NOT_CHECKED"
            if visual_state is None
            else "ABSENT" if visual_state == "desconhecido" else "PRESENT"
        )
        reason = "sem_evidencia_atual"
        if visual_state in CONFIRMED_STATES:
            if self.confirm_transition(visual_state, now, visual_confidence):
                reason = "state_detector"
            else:
                reason = "transicao_pendente"
        elif self.state in TERMINAL_STATES:
            # Nem ausência visual nem âncoras circulares do BlueStacks
            # podem substituir uma tela terminal confirmada.
            reason = "estado_terminal_preservado"
            self.confidence = self.evaluate_state_confidence(now)
        elif self.anchor_streak >= 2 and self.state != "combate":
            self._confirm("combate", now, 0.55)
            reason = "ancoras_heuristicas_incerto"
        elif self.preserve_last_state(now):
            reason = "contexto_de_combate_preservado"
            self.confidence = self.evaluate_state_confidence(now)
        else:
            self.state = "desconhecido"
            self.confidence = 0.0

        # Um quadro global sem classificação quebra uma transição pendente;
        # um quadro sem verificação global não é evidência negativa.
        if visual_state == "desconhecido":
            self._pending = None
            self._pending_votes = 0

        return {
            "name": self.state,
            "confidence": round(self.confidence, 3),
            "uncertain": reason
            in {
                "ancoras_heuristicas_incerto",
                "contexto_de_combate_preservado",
                "estado_terminal_preservado",
                "transicao_pendente",
                "sem_evidencia_atual",
            },
            "reason": reason,
            "evidence": {
                "visual": visual_observation,
                "hud": anchor_observation,
                "pending_transition": self._pending,
                "pending_votes": self._pending_votes,
            },
            "anchors": {
                "attack": bool(getattr(anchors, "attack", False)),
                "movement": bool(getattr(anchors, "movement", False)),
                "gadget": bool(getattr(anchors, "gadget", False)),
                "score": float(getattr(anchors, "score", 0.0)),
            },
        }

    def update(
        self,
        anchors: Any | None,
        now: float,
        visual_state: str | None = None,
        visual_confidence: float = 0.0,
    ) -> dict:
        """Compatibilidade com HUDContext.update() da V1."""
        return self.observe_state(anchors, now, visual_state, visual_confidence)
