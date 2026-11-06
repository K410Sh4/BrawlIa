"""Exemplo: python analyze.py minha_captura.png --output resultado"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
from brawlla.engine import SessionAnalyzer
from brawlla.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brawlla: análise experimental, somente leitura."
    )
    parser.add_argument("image", type=Path, help="Arquivo PNG/JPEG de uma partida")
    parser.add_argument(
        "--output", type=Path, default=Path("resultado"), help="Diretório de saída"
    )
    parser.add_argument("--config", type=Path, default=Path("configuracoes.json"))
    args = parser.parse_args()
    frame = cv2.imdecode(
        __import__("numpy").fromfile(str(args.image), dtype="uint8"), cv2.IMREAD_COLOR
    )
    if frame is None:
        parser.error("Não foi possível abrir esta imagem.")
    annotated, detections = SessionAnalyzer(load_config(args.config)).analyze(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    png_path = args.output / "analise.png"
    json_path = args.output / "deteccoes.json"
    ok, encoded = cv2.imencode(".png", annotated)
    if not ok:
        raise RuntimeError("Erro ao codificar PNG.")
    encoded.tofile(str(png_path))
    json_path.write_text(
        json.dumps(detections, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"PNG: {png_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")
    print(
        "Barras: verde=",
        bool(detections["player_health_bar_candidate"]),
        "| vermelhas=",
        len(detections["red_health_bar_candidates"]),
    )


if __name__ == "__main__":
    main()
