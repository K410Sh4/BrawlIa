"""Grava a análise ANOTADA em vídeo e JSONL, sem bloquear a captura.

Não executa comandos no jogo. Guarda o quadro mais recente (backpressure)
para impedir que uma fila de frames consuma memória ou gere vídeo atrasado.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading
import time

import cv2
import numpy as np
from .metrics import SessionMetrics
from .reporting import save_report
from .config import VERSION_SHORT, FPS_CHOICES
from .storage import new_session_id, write_json


def fit_video_frame(frame, width, height):
    """Encaixa no contêiner sem deformar quando a janela muda de proporção."""
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame, [0, 0, w, h]
    scale = min(width / w, height / h)
    dw, dh = min(width, max(1, round(w * scale))), min(height, max(1, round(h * scale)))
    left, top = (width - dw) // 2, (height - dh) // 2
    result = np.zeros((height, width, 3), dtype=frame.dtype)
    result[top : top + dh, left : left + dw] = cv2.resize(
        frame, (dw, dh), interpolation=cv2.INTER_AREA
    )
    return result, [left, top, dw, dh]


class AnalysisRecorder:
    def __init__(
        self,
        root: Path,
        fps: int = 15,
        max_minutes: int = 30,
        max_bytes: int = 2 * 1024**3,
        min_free_bytes: int = 512 * 1024**2,
        *,
        log_observations: bool = True,
    ):
        if fps not in FPS_CHOICES:
            raise ValueError("FPS do vídeo deve ser 5, 10, 15, 30 ou 60.")
        if max_minutes < 1 or max_bytes < 1 or min_free_bytes < 0:
            raise ValueError("Limites de gravação inválidos.")
        self.log_observations = log_observations
        self.metrics = SessionMetrics()
        self.fps = fps
        self.max_frames = fps * 60 * max_minutes
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.folder = (Path(root) / new_session_id()).resolve()
        self.verified = False
        self.video_path: Path | None = None
        self.log_path = self.folder / (
            "analise.jsonl" if log_observations else "video_frames.jsonl"
        )
        self.summary_path = self.folder / "resumo.json"
        self.written = 0
        self.records = 0
        self.dropped = 0
        self.downsampled = 0
        self.error: str | None = None
        self.limit_reached: str | None = None
        self._cond = threading.Condition()
        self._slot: tuple | None = None
        self._stop_requested = False
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Gravador já iniciado")
        self.folder.mkdir(parents=True, exist_ok=False)
        self.log_path.touch(exist_ok=False)
        self._thread = threading.Thread(
            target=self._write_loop, name="BrawllaVideoRecorder", daemon=False
        )
        self._thread.start()

    def submit(self, annotated: np.ndarray, detections: dict) -> None:
        if (
            not isinstance(annotated, np.ndarray)
            or annotated.ndim != 3
            or annotated.shape[2] != 3
        ):
            return
        capture = detections.get("capture", {})
        item = (
            detections.get("capture_timestamp_s", time.monotonic()),
            capture.get("captured_unix", time.time()),
            annotated,
            detections,
        )
        with self._cond:
            if self._stop_requested or self._done.is_set():
                return
            if self._slot is not None:
                self.dropped += 1
            self._slot = item
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stop_requested = True
            self._cond.notify_all()

    def join(self, timeout: float | None = None) -> bool:
        if self._thread is not None:
            self._thread.join(timeout)
        return self._done.is_set()

    def snapshot(self) -> dict:
        with self._cond:
            return {
                "video": str(self.video_path) if self.video_path else "",
                "folder": str(self.folder),
                "written": self.written,
                "records": self.records,
                "dropped": self.dropped,
                "downsampled": self.downsampled,
                "error": self.error,
                "limit_reached": self.limit_reached,
                "done": self._done.is_set(),
                "verified": self.verified,
            }

    def _open_writer(self, size: tuple[int, int]) -> cv2.VideoWriter:
        for filename, codec in (("analise.mp4", "mp4v"), ("analise.avi", "MJPG")):
            dest = self.folder / filename
            writer = cv2.VideoWriter(
                str(dest), cv2.VideoWriter_fourcc(*codec), self.fps, size
            )
            if writer.isOpened():
                self.video_path = dest
                return writer
            writer.release()
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
        raise RuntimeError("Nenhum codificador MP4/AVI disponível no OpenCV.")

    def _write_loop(self) -> None:
        writer: cv2.VideoWriter | None = None
        first_t: float | None = None
        last_frame = None
        last_frame_id = None
        width = height = 0
        stop_reason = "parada pelo usuário"
        try:
            with self.log_path.open("w", encoding="utf-8") as log:
                while True:
                    with self._cond:
                        self._cond.wait_for(
                            lambda: self._slot is not None or self._stop_requested
                        )
                        if self._slot is None and self._stop_requested:
                            break
                        item = self._slot
                        self._slot = None
                    if item is None:
                        continue
                    tmono, tunix, frame, detections = item
                    if first_t is None:
                        first_t = tmono
                        src_height, src_width = frame.shape[:2]
                        # A maioria dos codificadores de vídeo exige dimensões pares.
                        width = src_width + src_width % 2
                        height = src_height + src_height % 2
                        if shutil.disk_usage(self.folder).free < self.min_free_bytes:
                            stop_reason = "espaço livre insuficiente para começar"
                            self.limit_reached = stop_reason
                            break
                        writer = self._open_writer((width, height))
                    # Mesmo tamanho durante toda a sessão (janela pode ser redimensionada).
                    original_size = [int(frame.shape[1]), int(frame.shape[0])]
                    frame, content_rect = fit_video_frame(frame, width, height)
                    # Mapeamento dos timestamps para um vídeo de FPS constante.
                    index = max(0, int((tmono - first_t) * self.fps))
                    repeat_info = None
                    if index < self.written:
                        self.downsampled += 1
                        video_index = None
                    else:
                        # Repetições preservam a duração real caso a análise caia de FPS.
                        repeats = index - self.written
                        if repeats:
                            repeat_info = {
                                "start": self.written,
                                "count": min(repeats, self.max_frames - self.written),
                                "source_frame_id": last_frame_id,
                            }
                        for _ in range(min(repeats, self.max_frames - self.written)):
                            writer.write(
                                last_frame if last_frame is not None else frame
                            )
                            self.written += 1
                        video_index = None
                        if self.written < self.max_frames:
                            writer.write(frame)
                            video_index = self.written
                            self.written += 1
                        last_frame = frame
                        last_frame_id = detections.get("frame_id")
                    log.write(
                        json.dumps(
                            {
                                "timestamp_utc": datetime.fromtimestamp(
                                    tunix, timezone.utc
                                ).isoformat(),
                                "elapsed_s": round(tmono - first_t, 3),
                                "session_id": detections.get(
                                    "session_id", self.folder.name
                                ),
                                "frame_id": detections.get("frame_id"),
                                "capture_timestamp_s": detections.get(
                                    "capture_timestamp_s"
                                ),
                                "video_frame": video_index,
                                "video_repeats": repeat_info,
                                "input_resolution": original_size,
                                "video_content_xywh": content_rect,
                                **(
                                    {"detections": detections}
                                    if self.log_observations
                                    else {}
                                ),
                                "analysis_is_heuristic": True,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if self.log_observations:
                        self.metrics.observe(tmono - first_t, detections)
                    self.records += 1
                    if self.records % self.fps == 0:
                        log.flush()
                    if self.written >= self.max_frames:
                        stop_reason = "limite de duração atingido"
                    elif self.records % max(1, self.fps * 5) == 0:
                        video_size = (
                            self.video_path.stat().st_size
                            if self.video_path and self.video_path.exists()
                            else 0
                        )
                        if video_size + log.tell() >= self.max_bytes:
                            stop_reason = "limite de tamanho atingido"
                        elif shutil.disk_usage(self.folder).free < self.min_free_bytes:
                            stop_reason = "pouco espaço livre em disco"
                    if stop_reason != "parada pelo usuário":
                        self.limit_reached = stop_reason
                        with self._cond:
                            self._stop_requested = True
                            self._slot = None
                        break
        except Exception as exc:
            self.error = str(exc)
            stop_reason = "erro de gravação"
        finally:
            if writer is not None:
                try:
                    writer.release()  # Necessário para finalizar o contêiner MP4/AVI.
                except Exception as exc:
                    self.error = self.error or f"Falha ao finalizar vídeo: {exc}"
            # Verifica o contêiner depois de fechá-lo, em vez de assumir que write() gravou corretamente.
            if self.written:
                reader = None
                try:
                    if (
                        not self.video_path
                        or not self.video_path.exists()
                        or self.video_path.stat().st_size == 0
                    ):
                        raise RuntimeError(
                            "Arquivo de vídeo ausente ou vazio após finalizar"
                        )
                    reader = cv2.VideoCapture(str(self.video_path))
                    if not reader.isOpened() or not reader.read()[0]:
                        raise RuntimeError(
                            "Vídeo gerado não pôde ser reaberto para leitura"
                        )
                    self.verified = True
                except Exception as exc:
                    self.error = self.error or str(exc)
                finally:
                    if reader is not None:
                        reader.release()
            report_error = None
            try:
                if self.log_observations:
                    save_report(self.metrics.summary(), self.folder / "relatorio.json")
            except Exception as exc:
                report_error = str(exc)
                self.error = self.error or f"Falha ao gerar resumo: {exc}"
            summary = {
                "app_version": VERSION_SHORT,
                "read_only": True,
                "status": stop_reason,
                "video_file": self.video_path.name if self.video_path else None,
                "events_file": self.log_path.name,
                "fps": self.fps,
                "video_resolution": [width, height],
                "video_frames": self.written,
                "video_verified": self.verified,
                "events_recorded": self.records,
                "queued_frames_dropped": self.dropped,
                "events_not_written_to_video_due_to_fps": self.downsampled,
                "session_metrics": self.metrics.summary(),
                "error": self.error,
                "report_error": report_error,
                "report_file": (
                    "relatorio.html"
                    if (self.folder / "relatorio.html").exists()
                    else None
                ),
                "warning": "Detecções e rastreamento são heurísticos; não equivalem a dados rotulados.",
            }
            try:
                write_json(self.summary_path, summary)
            except Exception as exc:
                self.error = self.error or f"Falha ao salvar resumo: {exc}"
            finally:
                self._done.set()
