"""Dataset local assíncrono: imagens cruas, metadados e previsões separadas.

Amostras ainda não são rótulos verdadeiros. Escrita limitada, sem upload.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import queue
import shutil
import threading
import cv2
from .recognizer import RegionImage
from .dataset_curator import DatasetCurator


class DatasetStore:
    def __init__(
        self,
        folder,
        *,
        sample_interval_s=0.5,
        max_images=2000,
        max_bytes=512 * 1024**2,
        min_free_bytes=512 * 1024**2,
        queue_size=64,
    ):
        self.folder = Path(folder).resolve()
        self.sample_interval_s = sample_interval_s
        self.max_images = max_images
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.queue = queue.Queue(maxsize=queue_size)
        self.written = 0
        self.records = 0
        self.bytes_saved = 0
        self.dropped = 0
        self.error = None
        self.curator = DatasetCurator(interval_s=sample_interval_s)
        self.selected = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        (self.folder / "images").mkdir(parents=True, exist_ok=False)
        self._thread = threading.Thread(
            target=self._run, name="BrawlIAV1Dataset", daemon=False
        )
        self._thread.start()

    def submit(self, result: dict, regions: list[RegionImage]):
        if self._thread is None:
            raise RuntimeError("DatasetStore não iniciado")
        if self.error:
            raise RuntimeError(self.error)
        sample, reason = self.curator.select_samples(result)
        if sample:
            self.selected += 1
        samples = (
            [
                (r.name, r.rect.xywh(), r.image.copy(), r.captured_at, reason)
                for r in regions
            ]
            if sample else []
        )
        try:
            self.queue.put_nowait((dict(result), samples))
        except queue.Full:
            self.dropped += 1
            # Não ocultar perda de histórico.
            self.error = "Fila de dataset cheia, registros podem estar incompletos"
            raise RuntimeError(self.error)

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    def snapshot(self):
        return {
            "folder": str(self.folder),
            "written": self.written,
            "records": self.records,
            "bytes_saved": self.bytes_saved,
            "dropped": self.dropped,
            "selected_cycles": self.selected,
            "duplicates_skipped": self.curator.duplicates_skipped,
            "error": self.error,
        }

    def _run(self):
        try:
            with (
                (self.folder / "observations.jsonl").open(
                    "w", encoding="utf-8"
                ) as events,
                (self.folder / "manifest.jsonl").open(
                    "w", encoding="utf-8"
                ) as manifest,
            ):
                while not self._stop.is_set() or not self.queue.empty():
                    try:
                        result, samples = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    encoded_observation = json.dumps(result, ensure_ascii=False) + "\n"
                    observation_cost = len(encoded_observation.encode("utf-8"))
                    if self.bytes_saved + observation_cost > self.max_bytes:
                        self.error = "Limite do dataset atingido pelas observacoes"
                        break
                    events.write(encoded_observation)
                    self.bytes_saved += observation_cost
                    self.records += 1
                    for index, (name, coords, frame, captured_at, reason) in enumerate(samples):
                        if self.written >= self.max_images:
                            break
                        if self.records % 16 == 0 and (
                            shutil.disk_usage(self.folder).free < self.min_free_bytes
                        ):
                            self.error = "Pouco espaço livre, gravação encerrada"
                            break
                        duplicate_key = (
                            name, tuple(coords),
                            tuple(result["screen_resolution"]),
                            result.get("state", {}).get("name"),
                        )
                        if self.curator.detect_duplicate_samples(
                            frame, duplicate_key, force=reason == "critical_event"
                        ):
                            continue
                        success, png = cv2.imencode(
                            ".png",
                            frame,
                            [cv2.IMWRITE_PNG_COMPRESSION, 1],
                        )
                        if not success:
                            self.error = "Codificação PNG falhou"
                            break
                        filename = "images/frame_{:09d}_{:02d}.png".format(
                            result["frame_id"], index
                        )
                        sha256 = hashlib.sha256(png.tobytes()).hexdigest()
                        entry = {
                            "frame_id": result["frame_id"],
                            "timestamp_s": result["timestamp_s"],
                            "captured_at_s": captured_at,
                            "image": filename,
                            "sha256": sha256,
                            "width": frame.shape[1],
                            "height": frame.shape[0],
                            "screen_resolution": result["screen_resolution"],
                            "roi_xywh": coords,
                            "source_name": name,
                            "capture_mode": result["capture_mode"],
                            "sample_reason": reason,
                            "recognition_version": result.get("recognition_version"),
                            "label_status": "unreviewed",
                            "ground_truth": False,
                            "prediction": result["tracks"],
                        }
                        line = json.dumps(entry, ensure_ascii=False) + "\n"
                        cost = png.nbytes + len(line.encode("utf-8"))
                        if self.bytes_saved + cost > self.max_bytes:
                            self.error = "Limite de tamanho do dataset atingido"
                            break
                        destination = self.folder / filename
                        destination.write_bytes(png.tobytes())
                        manifest.write(line)
                        self.bytes_saved += cost
                        self.written += 1
                    if self.error:
                        break
                    if self.records % 10 == 0:
                        events.flush()
                        manifest.flush()
        except Exception as exc:
            self.error = self.error or str(exc)
        finally:
            (self.folder / "dataset_summary.json").write_text(
                json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
