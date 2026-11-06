"""Avaliação de localização de barras sobre PNGs com revisão humana."""

from collections import Counter
import json
from pathlib import Path
import time

import cv2

from .config import AnalyzerConfig
from .dataset import load_entries, validate_annotation
from .stream import resize_for_analysis
from .vision import detect


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return overlap / max(1e-12, aw * ah + bw * bh - overlap)


def match_boxes(predicted, truth, threshold=0.5):
    pairs = sorted(
        (
            (iou(a, b), i, j)
            for i, a in enumerate(predicted)
            for j, b in enumerate(truth)
        ),
        reverse=True,
    )
    seen_p = set()
    seen_t = set()
    for score, i, j in pairs:
        if score >= threshold and i not in seen_p and j not in seen_t:
            seen_p.add(i)
            seen_t.add(j)
    return len(seen_p), len(predicted) - len(seen_p), len(truth) - len(seen_t)


def reviewed_frames(root):
    root = Path(root)
    folders = (
        [root]
        if (root / "manifest.jsonl").exists()
        else sorted(p.parent for p in root.glob("*/manifest.jsonl"))
    )
    for folder in folders:
        for entry in load_entries(folder):
            path = folder / "labels" / (Path(entry["image"]).stem + ".json")
            if not path.exists():
                continue
            label = json.loads(path.read_text(encoding="utf-8"))
            validate_annotation(label, entry["width"], entry["height"])
            if (
                label["status"] not in ("accepted", "corrected")
                or label.get("reviewed_by") != "human"
            ):
                continue
            if (label.get("session_id"), label.get("frame_id"), label.get("image")) != (
                entry["session_id"],
                entry["frame_id"],
                entry["image"],
            ):
                raise ValueError("Rótulo pertence a outro quadro")
            frame = cv2.imread(str(folder / entry["image"]))
            if frame is None or list(frame.shape[:2]) != [
                entry["height"],
                entry["width"],
            ]:
                raise ValueError("Imagem ausente ou incompatível")
            yield entry, label, frame


def evaluate_dataset(root, config=None, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError("IoU deve estar entre 0 e 1")
    config = config or AnalyzerConfig()
    counts = Counter()
    times = []
    sessions = set()
    for entry, label, frame in reviewed_frames(root):
        image = resize_for_analysis(frame, config.vision.reference_width)
        started = time.perf_counter()
        data, _ = detect(image, config=config.vision)
        times.append((time.perf_counter() - started) * 1000)
        sx, sy = frame.shape[1] / image.shape[1], frame.shape[0] / image.shape[0]
        predicted = []
        for candidate in data["health_bar_candidates"]:
            box = [
                candidate["x"] * sx,
                candidate["y"] * sy,
                candidate["width"] * sx,
                candidate["height"] * sy,
            ]
            if not any(iou(box, existing) > 0.7 for existing in predicted):
                predicted.append(box)
        truth = [
            box["bbox"] for box in label["boxes"] if box["category"] == "barra_vida"
        ]
        tp, fp, fn = match_boxes(predicted, truth, threshold)
        counts.update(frames=1, tp=tp, fp=fp, fn=fn)
        sessions.add(entry["session_id"])
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return {
        "frames": counts["frames"],
        "sessions": sorted(sessions),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "mean_visual_ms": sum(times) / len(times) if times else None,
        "iou_threshold": threshold,
        "limitations": [
            "Avalia somente localização de barras em PNGs revisados, sem estado de partida, movimento temporal, dono ou leitura numérica de HP.",
            "Cada amostra é independente; não substitui teste de sequências completas nem medição ao vivo.",
        ],
    }
