"""Histórico completo de análises em fila própria; nunca substitui eventos aceitos."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import shutil
import threading
import time

from .metrics import SessionMetrics
from .reporting import save_report


class ObservationRecorder:
    def __init__(
        self,
        folder,
        *,
        video_enabled=False,
        queue_capacity=256,
        max_bytes=256 * 1024**2,
        min_free_bytes=512 * 1024**2,
    ):
        if queue_capacity < 1 or max_bytes < 1 or min_free_bytes < 0:
            raise ValueError("Limites de histórico inválidos")
        self.folder = Path(folder).resolve()
        self.log_path = self.folder / "analise.jsonl"
        self.video_enabled = video_enabled
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.queue = queue.Queue(maxsize=queue_capacity)
        self.records = self.accepted = self.rejected = self.bytes_saved = 0
        self.error = ""
        self.metrics = SessionMetrics()
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._first_time = None

    def start(self):
        self.log_path.touch(exist_ok=False)
        self._thread = threading.Thread(
            target=self._run, name="BrawllaHistory", daemon=False
        )
        self._thread.start()

    def submit(self, data):
        with self._lock:
            if self.error or self._stop.is_set():
                raise RuntimeError(self.error or "Histórico já encerrado")
            try:
                self.queue.put_nowait(deepcopy(data))
            except queue.Full:
                self.rejected += 1
                self.error = "Fila do histórico cheia; leitura interrompida para evitar perda silenciosa."
                self._stop.set()
                raise RuntimeError(self.error)
            self.accepted += 1

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    def snapshot(self):
        with self._lock:
            return {
                "folder": str(self.folder),
                "records": self.records,
                "accepted": self.accepted,
                "rejected": self.rejected,
                "bytes_saved": self.bytes_saved,
                "error": self.error,
                "queue_capacity": self.queue.maxsize,
                "pending": self.queue.qsize(),
                "complete": self._stop.is_set()
                and self.records == self.accepted
                and not self.error,
            }

    def _run(self):
        try:
            with self.log_path.open("w", encoding="utf-8") as log:
                while not self._stop.is_set() or not self.queue.empty():
                    try:
                        data = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    captured = data.get("capture_timestamp_s", time.monotonic())
                    if self._first_time is None:
                        self._first_time = captured
                    wall = data.get("capture", {}).get("captured_unix", time.time())
                    entry = {
                        "timestamp_utc": datetime.fromtimestamp(
                            wall, timezone.utc
                        ).isoformat(),
                        "elapsed_s": round(captured - self._first_time, 6),
                        "session_id": data.get("session_id", self.folder.name),
                        "frame_id": data.get("frame_id"),
                        "capture_timestamp_s": captured,
                        "video_frame": None,
                        "video_repeats": None,
                        "video_mapping": (
                            "video_frames.jsonl" if self.video_enabled else None
                        ),
                        "input_resolution": data.get("resolution"),
                        "detections": data,
                        "analysis_is_heuristic": True,
                    }
                    line = json.dumps(entry, ensure_ascii=False) + "\n"
                    cost = len(line.encode("utf-8"))
                    if self.bytes_saved + cost > self.max_bytes:
                        raise RuntimeError(
                            "Limite de bytes do histórico atingido; leitura deve ser encerrada."
                        )
                    if (
                        self.records % 32 == 0
                        and shutil.disk_usage(self.folder).free - cost
                        < self.min_free_bytes
                    ):
                        raise RuntimeError(
                            "Espaço livre insuficiente para o histórico."
                        )
                    log.write(line)
                    log.flush()
                    self.metrics.observe(captured - self._first_time, data)
                    with self._lock:
                        self.records += 1
                        self.bytes_saved += cost
        except Exception as exc:
            with self._lock:
                self.error = str(exc)
            self._stop.set()
        finally:
            try:
                save_report(self.metrics.summary(), self.folder / "relatorio.json")
            except Exception as exc:
                with self._lock:
                    self.error = self.error or f"Falha no relatório do histórico: {exc}"
