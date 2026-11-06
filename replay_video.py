"""Reprocessa um vídeo local e gera um relatório; não controla o emulador."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import cv2
from brawlla.engine import SessionAnalyzer
from brawlla.config import load_config
from brawlla.metrics import SessionMetrics
from brawlla.reporting import save_report


def replay(
    source: Path, output: Path, annotated_input: bool = False, config=None
) -> dict:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Não foi possível abrir o vídeo.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not 0 < fps < 1000:
        capture.release()
        raise ValueError(
            "Vídeo sem FPS válido; não é possível preservar a linha do tempo."
        )
    analyzer = SessionAnalyzer(config)
    metrics = SessionMetrics()
    started = time.perf_counter()
    index = 0
    try:
        output.mkdir(parents=True, exist_ok=True)
        with (output / "replay.jsonl").open("w", encoding="utf-8") as log:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                elapsed = index / fps
                _, data = analyzer.analyze(frame, timestamp_s=elapsed)
                metrics.observe(elapsed, data)
                log.write(
                    json.dumps(
                        {
                            "elapsed_s": round(elapsed, 6),
                            "video_frame": index,
                            "detections": data,
                            "analysis_is_heuristic": True,
                            "source_is_annotated": annotated_input,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                index += 1
        if index == 0:
            raise ValueError("O vídeo não contém quadros decodificáveis.")
        result = metrics.summary()
        result.update(
            {
                "source_is_annotated": annotated_input,
                "video_fps": fps,
                "replay_wall_time_s": round(time.perf_counter() - started, 3),
                "replay_complete_to_decoder_eof": True,
            }
        )
        result["limitations"].extend(
            [
                "Replay offline: FPS do vídeo não equivale a FPS de processamento ao vivo.",
                "Vídeos anotados contêm cores e contornos artificiais que podem alterar as detecções.",
                "Não mede acurácia nem descarta a necessidade de validação em capturas brutas separadas.",
            ]
        )
        save_report(result, output / "relatorio.json")
        return result
    finally:
        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BrawlIA: replay local com relatório JSON/HTML."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configuracoes.json"))
    parser.add_argument("--output", type=Path, default=Path("resultado_replay"))
    parser.add_argument(
        "--annotated-input",
        action="store_true",
        help="Informe quando o vídeo já contém marcações.",
    )
    args = parser.parse_args()
    try:
        result = replay(
            args.video, args.output, args.annotated_input, load_config(args.config)
        )
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f"Erro: {exc}\n")
    print(
        f"{result['analyzed_frames']} quadros analisados. Relatório: {(args.output/'relatorio.html').resolve()}"
    )
