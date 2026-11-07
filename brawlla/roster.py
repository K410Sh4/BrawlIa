"""Identidades da apresentação, separadas dos IDs temporários do tracker.

OCR só recebe pequenos recortes em thread daemon, nunca bloqueia a
análise. Nomes pouco confiáveis não geram vínculos automáticos.
"""

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from queue import Empty, Full, Queue
import threading
import unicodedata

import cv2
import numpy as np


def normalize_name(text):
    return "".join(
        c for c in unicodedata.normalize("NFKC", str(text)).casefold() if c.isalnum()
    )


def name_regions(frame):
    height, width = frame.shape[:2]
    return [
        (
            round((col / 5 + 0.006) * width),
            round(ystart * height),
            round(((col + 1) / 5 - 0.006) * width),
            round(yend * height),
        )
        for ystart, yend in ((0.405, 0.463), (0.772, 0.835))
        for col in range(5)
    ]


def roster_visible(frame):
    """Detecta as duas fileiras de placas escuras, sem inferência neural."""
    if frame.ndim != 3 or frame.shape[0] < 400 or frame.shape[1] < 700:
        return False
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    count = 0
    for x1, y1, x2, y2 in name_regions(frame):
        patch = grayscale[y1 + (y2 - y1) // 3 : y2, x1:x2]
        if patch.size and np.count_nonzero(patch < 75) >= patch.size * 0.4:
            count += 1
    return count >= 9


def crop_text(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(
        frame[y1:y2, x1:x2], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )


def read_name(engine, image):
    result = engine(image)
    options = [
        (str(text).strip(), float(score))
        for text, score in zip(
            getattr(result, "txts", ()) or (), getattr(result, "scores", ()) or ()
        )
        if len(normalize_name(text)) >= 4 and 0 <= float(score) <= 1
    ]
    return max(options, key=lambda item: (item[1], len(item[0])), default=("", 0.0))


class MatchRoster:
    """Relação da partida: participant_id fixo, track_id descartável."""

    def __init__(self, config, *, ocr_factory=None):
        self.config = config
        self.ocr_factory = ocr_factory
        self.jobs = Queue(maxsize=3)
        self.completed = Queue()
        self.worker = None
        self.engine = None
        self.generation = 0
        self.reset()

    def reset(self):
        self.generation += 1
        self.detected = False
        self.had_combat = False
        self.attempts = 0
        self.last_roster = float("-inf")
        self.last_combat = {}
        self.last_any_name = float("-inf")
        self.next_kind = "jogador_candidato"
        self.error = None
        self.local_id = None
        self.bindings = {}
        self.active_track_ids = set()
        self.votes = [Counter() for _ in range(10)]
        self.track_votes = defaultdict(Counter)
        self.participants = [
            dict(
                participant_id=f"P{i+1:02d}",
                slot=i + 1,
                name=None,
                status="pending",
                confidence=0.0,
                is_local=False,
            )
            for i in range(10)
        ]

    def _send(self, kind, payload):
        if self.error:
            return False
        try:
            self.jobs.put_nowait((self.generation, kind, payload))
        except Full:
            return False
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(
                target=self._run, name="BrawlIARosterOCR", daemon=True
            )
            self.worker.start()
        return True

    def _run(self):
        try:
            if self.engine is None:
                if self.ocr_factory is not None:
                    self.engine = self.ocr_factory()
                else:
                    from rapidocr import RapidOCR

                    self.engine = RapidOCR()
            while True:
                try:
                    generation, kind, payload = self.jobs.get(timeout=5)
                except Empty:
                    return
                try:
                    if kind == "roster":
                        answer = [read_name(self.engine, crop) for crop in payload]
                    else:
                        track_id, crop = payload
                        answer = (track_id, *read_name(self.engine, crop))
                    self.completed.put((generation, kind, answer))
                except Exception as exc:
                    self.completed.put((generation, "error", type(exc).__name__))
                finally:
                    self.jobs.task_done()
        except ImportError:
            self.completed.put((self.generation, "error", "rapidocr_indisponivel"))
        except Exception as exc:
            self.completed.put((self.generation, "error", type(exc).__name__))

    def _poll(self):
        while True:
            try:
                generation, kind, payload = self.completed.get_nowait()
            except Empty:
                break
            if generation != self.generation:
                continue
            if kind == "error":
                self.error = payload
            elif kind == "roster":
                for slot, (name, confidence) in enumerate(payload):
                    if confidence < self.config.ocr_confidence:
                        continue
                    key = normalize_name(name)
                    if len(key) < 4:
                        continue
                    self.votes[slot][key] += 1
                    winner, votes = self.votes[slot].most_common(1)[0]
                    if winner == key:
                        item = self.participants[slot]
                        item["name"] = name
                        item["confidence"] = round(confidence, 3)
                        item["status"] = (
                            "confirmed"
                            if votes >= self.config.roster_votes
                            else "tentative"
                        )
            elif kind == "combat":
                tid, name, confidence = payload
                self._match_track(tid, name, confidence)

    def _match_track(self, track_id, text, confidence):
        if confidence < self.config.ocr_confidence:
            return
        name = normalize_name(text)
        if len(name) < 4:
            return
        candidates = sorted(
            (
                SequenceMatcher(None, name, normalize_name(p["name"])).ratio(),
                p["participant_id"],
            )
            for p in self.participants
            if p["status"] == "confirmed" and p["name"]
        )
        candidates.reverse()
        if not candidates or candidates[0][0] < self.config.match_threshold:
            return
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.12:
            return
        winner = candidates[0][1]
        self.track_votes[track_id][winner] += 1
        if (
            self.track_votes[track_id][winner] < self.config.match_votes
            or len(self.track_votes[track_id]) != 1
        ):
            return
        previous = next(
            (tid for tid, pid in self.bindings.items() if pid == winner), None
        )
        if previous is not None:
            if previous in self.active_track_ids:
                return
            del self.bindings[previous]
        self.bindings[track_id] = winner

    def observe(self, frame, state, timestamp, objects):
        """Não aguarda OCR; publica somente identidades efetivamente confirmadas."""
        self._poll()
        if state == "combate":
            self.had_combat = True
        elif roster_visible(frame):
            if self.had_combat:
                self.reset()  # Novo conjunto de jogadores após outra partida.
            self.detected = True
            if (
                self.attempts < self.config.roster_frames
                and timestamp - self.last_roster >= self.config.roster_interval_s
            ):
                crops = [crop_text(frame, r) for r in name_regions(frame)]
                if all(crop is not None for crop in crops) and self._send(
                    "roster", crops
                ):
                    self.attempts += 1
                    self.last_roster = timestamp

        if (
            self.detected
            and state == "combate"
            and any(p["status"] == "confirmed" for p in self.participants)
            and timestamp - self.last_any_name >= self.config.combat_interval_s
        ):
            # Um recorte por janela de tempo. Alterna prioridade entre
            # jogador e outros personagens, sem sobrecarregar o FPS.
            kinds = (
                ("jogador_candidato", "personagem_possivel")
                if self.next_kind == "jogador_candidato"
                else ("personagem_possivel", "jogador_candidato")
            )
            for kind in kinds:
                eligible = sorted(
                    (
                        obj
                        for obj in objects
                        if obj.get("kind") == kind
                        and isinstance(obj.get("track_id"), int)
                        and obj.get("track_id") not in self.bindings
                        and obj.get("track_hits", 0)
                        >= (5 if kind == "personagem_possivel" else 1)
                    ),
                    key=lambda obj: -obj.get("track_hits", 0),
                )
                if not eligible:
                    continue
                for obj in eligible:
                    tid = obj["track_id"]
                    if timestamp - self.last_combat.get(tid, float("-inf")) < 1.0:
                        continue
                    region = obj.get("name_region")
                    if isinstance(region, (tuple, list)) and len(region) == 4:
                        x, y, w, h = region
                        box = (x - 5, y - 5, x + w + 5, y + h + 5)
                    elif kind == "personagem_possivel":
                        # Hipótese de posição do nome acima da barra.
                        # Uma ausência de OCR não gera identidade.
                        x, y, w = (obj.get(k) for k in ("x", "y", "width"))
                        if not all(isinstance(n, (int, float)) for n in (x, y, w)):
                            continue
                        box = (x - 35, y - 32, x + w + 35, y - 4)
                    else:
                        continue
                    crop = crop_text(frame, box)
                    if crop is not None and self._send("combat", (tid, crop)):
                        self.last_combat[tid] = timestamp
                        self.last_any_name = timestamp
                        self.next_kind = (
                            "personagem_possivel"
                            if kind == "jogador_candidato"
                            else "jogador_candidato"
                        )
                        break
                if self.last_any_name == timestamp:
                    break

        for obj in objects:
            pid = self.bindings.get(obj.get("track_id"))
            obj["participant_id"] = pid
            obj["participant_identity_status"] = "matched_ocr" if pid else "unresolved"
            if pid and obj.get("kind") == "jogador_candidato":
                self.local_id = pid
        self.active_track_ids = {
            obj.get("track_id")
            for obj in objects
            if isinstance(obj.get("track_id"), int)
        }
        for item in self.participants:
            item["is_local"] = item["participant_id"] == self.local_id
        return self.snapshot()

    def snapshot(self):
        return {
            "status": (
                "ocr_unavailable"
                if self.error
                else "detected" if self.detected else "not_seen"
            ),
            "ocr_error": self.error,
            "local_participant_id": self.local_id,
            "confirmed_names": sum(
                p["status"] == "confirmed" for p in self.participants
            ),
            "bound_tracks": len(self.bindings),
            "participants": (
                [item.copy() for item in self.participants] if self.detected else []
            ),
        }
