"""Runtime observador V1; captura e análise no mesmo thread para MSS/GDI.

O modo completo da V1.4 permanece disponível como fallback separado.
"""
from __future__ import annotations
import argparse
from collections import deque
from pathlib import Path
import time
import cv2
import numpy as np

from brawlla.capture import AutoWindowCapture, enable_dpi_awareness
from brawlla.storage import user_data_directory, new_session_id
from .geometry import Rect
from .recognizer import RecognitionV1, RegionImage
from .roi_manager import ROIManager
from .dataset_store import DatasetStore


def _preview(regions, size, result, full):
    if full:
        image = regions[0].image.copy()
    else:
        image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.putText(image, "REGIONAL: pixels fora das ROIs nao observados",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (220, 220, 220), 2)
        for region in regions:
            r = region.rect
            image[r.y:r.y + r.h, r.x:r.x + r.w] = region.image
    for track in result["tracks"]:
        x, y, w, h = track["xywh"]
        cv2.rectangle(image, (x, y), (x + w, y + h),
                      (0, 220, 100) if track["status"] == "observado"
                      else (0, 140, 230), 2)
        cv2.putText(image, "#" + str(track["track_id"]) + " " + track["status"],
                    (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (225, 235, 245), 1)
    limit = 960
    if image.shape[1] > limit:
        h = round(image.shape[0] * limit / image.shape[1])
        image = cv2.resize(image, (limit, h), interpolation=cv2.INTER_AREA)
    return image


def run(*, target_fps: int = 30, duration_s: float = 0,
        preview: bool = False, dataset_root: Path | None = None):
    if target_fps not in (5, 10, 15, 30, 60):
        raise ValueError("FPS deve ser 5/10/15/30/60")
    enable_dpi_awareness()
    root = (dataset_root or (
        user_data_directory() / "datasets" / "recognition_v1"
    )).resolve() / new_session_id()
    store = DatasetStore(root)
    capture = None
    starts = deque()
    samples = 0
    first_at = time.perf_counter()
    last_report = first_at
    detector = RecognitionV1()
    policy = ROIManager()
    screen_size = None
    error = None
    store.start()
    try:
        capture = AutoWindowCapture()
        print("Reconhecimento V1 iniciado; armazenamento local:", root, flush=True)
        while duration_s <= 0 or time.perf_counter() - first_at < duration_s:
            started = time.perf_counter()
            tracks = (
                detector.last_result.get("tracks", [])
                if detector.last_result else []
            )
            state = (
                detector.last_result.get("state", {}).get("name", "desconhecido")
                if detector.last_result else "desconhecido"
            )
            if screen_size is None:
                full = True
                reason = "primeiro_quadro"
            else:
                plan = policy.plan(started, screen_size, state, tracks)
                full, reason = plan.full, plan.reason
            if full:
                frame = capture.grab()
                height, width = frame.shape[:2]
                if screen_size and screen_size != (width, height):
                    detector.memory.reset("resolucao_alterada")
                screen_size = width, height
                regions = [RegionImage(
                    "full", Rect(0, 0, width, height), frame
                )]
                policy.note_full(started)
            else:
                regions = []
                try:
                    for region in plan.regions:
                        image = capture.grab_region(region.rect.xywh())
                        regions.append(RegionImage(
                            region.name, region.rect, image
                        ))
                except (RuntimeError, OSError, ValueError):
                    policy.request_full_scan("falha_de_captura_regional")
                    frame = capture.grab()
                    height, width = frame.shape[:2]
                    screen_size = width, height
                    regions = [RegionImage(
                        "full", Rect(0, 0, width, height), frame
                    )]
                    policy.note_full(started)
                    full = True
                    reason = "fallback_global"
            result = detector.observe(
                regions, screen_size, now=started, full=full
            )
            result["capture_reason"] = reason
            result["capture_backend"] = capture.backend
            result["acquired_pixels"] = sum(r.rect.area for r in regions)
            result["full_screen_pixels"] = screen_size[0] * screen_size[1]
            result["processing_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            store.submit(result, regions)
            samples += 1
            now = time.perf_counter()
            starts.append(now)
            while starts and now - starts[0] > 2:
                starts.popleft()
            if now - last_report >= 2:
                effective_fps = len(starts) / min(2, now - first_at)
                print(
                    "FPS {:.1f} | {} | {} alvos | {} ROIs | {:.0f}% pixels | {} PNGs".format(
                        effective_fps, result["state"]["name"],
                        len(result["tracks"]), len(regions),
                        100 * result["acquired_pixels"] /
                        result["full_screen_pixels"], store.written
                    ),
                    flush=True,
                )
                last_report = now
            if preview:
                cv2.imshow("BrawlIA Recognition V1 (Q para sair)",
                           _preview(regions, screen_size, result, full))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            remaining = 1.0 / target_fps - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        error = str(exc)
        print("ERRO:", error, flush=True)
        raise
    finally:
        if capture is not None:
            capture.close()
        if preview:
            cv2.destroyAllWindows()
        store.stop()
        store.join()
        print("Sessão encerrada:", store.snapshot(), flush=True)
        if error:
            print("Falha de execução:", error, flush=True)
    return store.snapshot()


def main():
    parser = argparse.ArgumentParser(
        description="BrawlIA V1: identificação regional e dataset local"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    run(
        target_fps=args.fps,
        duration_s=args.duration,
        preview=args.preview,
        dataset_root=args.dataset_root,
    )


if __name__ == "__main__":
    main()
