"""Geometria, qualidade e revisão de dados; independente da interface."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from .storage import write_json, utc_now

CATEGORIES = ("jogador", "outro_personagem", "caixa", "barra_vida")
STATUSES = ("unreviewed", "accepted", "corrected", "rejected", "uncertain")


def crop_frame(frame, roi=None, source_aspect=0):
    h, w = frame.shape[:2]
    if source_aspect and abs(w / h / source_aspect - 1) > 0.02:
        raise ValueError("Proporção da fonte mudou; calibre novamente o recorte.")
    x, y, rw, rh = roi or (0, 0, 1, 1)
    if (
        not all(math.isfinite(v) for v in (x, y, rw, rh))
        or min(x, y) < 0
        or min(rw, rh) <= 0
        or x + rw > 1.000001
        or y + rh > 1.000001
    ):
        raise ValueError("Recorte inválido")
    left, top = round(x * w), round(y * h)
    right, bottom = min(w, round((x + rw) * w)), min(h, round((y + rh) * h))
    if right - left < 320 or bottom - top < 200:
        raise ValueError("Recorte deve conter pelo menos 320 × 200 pixels.")
    return frame[top:bottom, left:right].copy(), {
        "source_resolution": [w, h],
        "roi_xywh": [left, top, right - left, bottom - top],
        "roi_normalized": list(roi or (0, 0, 1, 1)),
    }


class QualityMonitor:
    """Sinais visuais heurísticos: imagem estática não prova congelamento."""

    def __init__(self):
        self.previous = None
        self.previous_size = None
        self.static_since = None

    def inspect(self, frame, now):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(cv2.resize(frame, (96, 54)), cv2.COLOR_BGR2GRAY)
        flags = []
        if float(np.mean(gray <= 5)) > 0.99:
            flags.append("black_frame")
        if (
            self.previous is not None
            and abs(float(np.mean(gray)) - float(np.mean(self.previous))) > 40
        ):
            flags.append("possible_flash_or_scene_change")
        if self.previous_size and self.previous_size != [w, h]:
            flags.append("source_resized")
        if (
            self.previous is not None
            and float(np.mean(cv2.absdiff(gray, self.previous))) < 0.25
        ):
            if self.static_since is None:
                self.static_since = now
            if now - self.static_since >= 2:
                flags.append("possible_freeze_or_static_scene")
        else:
            self.static_since = None
        self.previous, self.previous_size = gray, [w, h]
        return {
            "flags": flags,
            "usable": "black_frame" not in flags,
            "occlusion_detection": "not_reliable_keep_window_uncovered",
        }


def proposed_boxes(data, width, height):
    """Somente caixas observadas, sem inventar corpo a partir da barra."""
    aw, ah = data.get("resolution", [width, height])
    sx, sy = width / aw, height / ah
    candidates = []
    player = data.get("player_health_bar_candidate")
    if player:
        candidates.append(player)
    candidates.extend(
        data.get(
            "other_health_bar_candidates", data.get("red_health_bar_candidates", [])
        )
    )
    candidates.extend(data.get("other_green_health_bar_candidates", []))
    boxes = []
    for item in candidates:
        if all(k in item for k in ("x", "y", "width", "height")):
            x, y = max(0, item["x"] * sx), max(0, item["y"] * sy)
            bw = min(width, (item["x"] + item["width"]) * sx) - x
            bh = min(height, (item["y"] + item["height"]) * sy) - y
            if bw > 0 and bh > 0:
                boxes.append(
                    {
                        "category": "barra_vida",
                        "bbox": [x, y, bw, bh],
                        "origin": "heuristic",
                    }
                )
    return boxes


def validate_annotation(value, width, height):
    if value.get("status") not in STATUSES:
        raise ValueError("Estado de revisão inválido")
    boxes = value.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("Lista de caixas necessária")
    for box in boxes:
        if box.get("category") not in CATEGORIES:
            raise ValueError("Categoria inválida")
        coords = box.get("bbox", [])
        if len(coords) != 4 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in coords
        ):
            raise ValueError("Caixa inválida")
        x, y, w, h = coords
        if (
            min(x, y) < 0
            or min(w, h) <= 0
            or x + w > width + 0.01
            or y + h > height + 0.01
        ):
            raise ValueError("Caixa fora da imagem")
    entities = {}
    for box in boxes:
        entity = box.get("entity_id")
        if entity is not None:
            if (
                box["category"] == "barra_vida"
                or not isinstance(entity, str)
                or not entity
                or len(entity) > 128
                or entity in entities
            ):
                raise ValueError("Identificador de corpo inválido ou duplicado")
            entities[entity] = box
    for box in boxes:
        owner = box.get("owner_id")
        if owner is not None and (
            box["category"] != "barra_vida"
            or not isinstance(owner, str)
            or owner not in entities
        ):
            raise ValueError("Dono da barra não corresponde a um corpo desta imagem")
    if (
        value.get("status") in ("accepted", "corrected")
        and not boxes
        and value.get("negative") is not True
    ):
        raise ValueError(
            "Marque explicitamente exemplo negativo para aprovar sem caixas."
        )
    if value.get("negative") and boxes:
        raise ValueError("Exemplo negativo deve estar sem caixas.")


def save_annotation(folder, entry, value):
    validate_annotation(value, entry["width"], entry["height"])
    write_json(
        Path(folder) / "labels" / (Path(entry["image"]).stem + ".json"),
        {
            **value,
            "session_id": entry["session_id"],
            "frame_id": entry["frame_id"],
            "image": entry["image"],
            "coordinate_space": "raw_roi_pixels",
            "reviewed_by": "human",
            "reviewed_utc": utc_now(),
        },
    )


def load_entries(folder):
    folder = Path(folder)
    entries = [
        json.loads(line)
        for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for entry in entries:
        if not all(
            key in entry
            for key in (
                "session_id",
                "frame_id",
                "prediction",
                "width",
                "height",
                "image",
            )
        ):
            raise ValueError(
                "Amostras antigas sem sincronização V0.8; colete uma nova sessão para revisar."
            )
        # Manifestos locais ainda não podem navegar para fora da pasta escolhida.
        prediction_path = (folder / entry["prediction"]).resolve()
        if (
            prediction_path.parent != folder.resolve()
            or not prediction_path.name.endswith(".prediction.json")
        ):
            raise ValueError("Caminho de previsão inválido no manifesto")
        path = (folder / entry["image"]).resolve()
        if path.parent != folder.resolve() or path.suffix != ".png":
            raise ValueError("Caminho de imagem inválido no manifesto")
    return entries


def export_dataset(root, destination):
    """COCO por sessão completa; somente revisão humana aceita/corrigida."""
    root, destination = Path(root), Path(destination)
    sessions = sorted(p.parent for p in root.glob("*/manifest.jsonl"))
    if (root / "manifest.jsonl").exists():
        sessions = [root]
    output = {
        split: {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i + 1, "name": name} for i, name in enumerate(CATEGORIES)
            ],
        }
        for split in ("train", "val", "test")
    }
    relationships = []
    unlinked_bars = 0
    assignments = {}
    legacy_sessions = []
    image_id = annotation_id = 0
    for folder in sessions:
        raw_entries = [
            json.loads(line)
            for line in (folder / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if raw_entries and all("frame_id" not in entry for entry in raw_entries):
            legacy_sessions.append(folder.name)
            continue
        for entry in load_entries(folder):
            label = folder / "labels" / (Path(entry["image"]).stem + ".json")
            if not label.exists():
                continue
            value = json.loads(label.read_text(encoding="utf-8"))
            validate_annotation(value, entry["width"], entry["height"])
            if (
                value.get("session_id") != entry["session_id"]
                or value.get("frame_id") != entry["frame_id"]
                or value.get("image") != entry["image"]
            ):
                raise ValueError("Rótulo pertence a outro quadro")
            if (
                value["status"] not in ("accepted", "corrected")
                or value.get("reviewed_by") != "human"
            ):
                continue
            image = cv2.imread(str(folder / entry["image"]))
            if image is None or list(image.shape[:2]) != [
                entry["height"],
                entry["width"],
            ]:
                raise ValueError(
                    "Imagem ausente ou dimensão incompatível: " + entry["image"]
                )
            # Hash estável por sessão: novas exportações não movem frames entre conjuntos.
            sid = entry["session_id"]
            bucket = int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) % 100
            split = "train" if bucket < 80 else "val" if bucket < 90 else "test"
            assignments[sid] = split
            image_id += 1
            relative = (
                (folder / entry["image"])
                .resolve()
                .relative_to(root.resolve())
                .as_posix()
            )
            output[split]["images"].append(
                {
                    "id": image_id,
                    "file_name": relative,
                    "width": entry["width"],
                    "height": entry["height"],
                    "session_id": sid,
                    "frame_id": entry["frame_id"],
                }
            )
            ids = {
                box["entity_id"]: annotation_id + i + 1
                for i, box in enumerate(value["boxes"])
                if "entity_id" in box
            }
            for box in value["boxes"]:
                annotation_id += 1
                if box["category"] == "barra_vida":
                    if box.get("owner_id"):
                        relationships.append(
                            {
                                "split": split,
                                "image_id": image_id,
                                "bar_annotation_id": annotation_id,
                                "owner_annotation_id": ids[box["owner_id"]],
                            }
                        )
                    else:
                        unlinked_bars += 1
                x, y, w, h = box["bbox"]
                output[split]["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": CATEGORIES.index(box["category"]) + 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
    for split, value in output.items():
        write_json(destination / (split + ".json"), value)
    write_json(destination / "relationships.json", relationships)
    summary = {
        "relationships": len(relationships),
        "unlinked_bars": unlinked_bars,
        "empty_splits": [key for key, value in output.items() if not value["images"]],
        "sessions": assignments,
        "legacy_sessions_skipped": legacy_sessions,
        "images": image_id,
        "annotations": annotation_id,
        "image_root": str(root.resolve()),
        "split_policy": "sha256_session_80_10_10",
        "warning": "Poucas sessões podem deixar conjuntos vazios. Não misture frames da mesma partida entre sessões.",
    }
    write_json(destination / "dataset.json", summary)
    return summary
