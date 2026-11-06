"""Compara apenas a política temporal com scores já gravados, sem reanalisar pixels."""

import argparse
import json
from pathlib import Path
from brawlla.config import load_config
from brawlla.state import StateStabilizer


def validate(path, config=None):
    policy = StateStabilizer(config)
    timeline = []
    previous = None
    window_start = window_end = None
    compared = []
    with Path(path).open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            old = row["detections"]["game_state"]
            timestamp = row["elapsed_s"]
            hud, defeat, results, menu = (
                old.get(key, 0)
                for key in ("hud_score", "defeat_score", "results_score", "menu_score")
            )
            cfg = policy.config
            if results >= cfg.results_threshold:
                raw, confidence = "resultados", results
            elif defeat >= cfg.defeat_threshold:
                raw, confidence = "derrota", defeat
            elif menu >= cfg.menu_threshold:
                raw, confidence = "pre_partida", menu
            elif hud >= cfg.hud_threshold:
                raw, confidence = "combate", hud
            else:
                raw, confidence = "desconhecido", max(hud, defeat, results, menu)
            new = policy.update(
                raw,
                confidence,
                timestamp,
                evidence_checked=not old.get("cached", False),
            )
            if new["name"] != previous:
                timeline.append(
                    {
                        "elapsed_s": timestamp,
                        "state": new["name"],
                        "reason": new["reason"],
                    }
                )
                previous = new["name"]
            if window_start is None and old["name"] == "combate":
                window_start = timestamp
            if window_start is not None and window_end is None:
                if old["name"] in ("derrota", "resultados"):
                    window_end = timestamp
                else:
                    compared.append((old["name"], new["name"]))
    return {
        "method": "Política temporal aplicada aos mesmos scores e timestamps; sem reprocessar pixels. Janela da primeira entrada em combate até o primeiro estado terminal do registro original.",
        "window_s": [window_start, window_end],
        "window_events": len(compared),
        "baseline_noncombat_events": sum(old != "combate" for old, new in compared),
        "v07_noncombat_events": sum(new != "combate" for old, new in compared),
        "new_timeline": timeline,
        "limitations": [
            "Testa continuidade do estado; não mede presença de jogador ou acurácia.",
            "Menu só é testável se menu_score existir no registro.",
            "Não recupera observações que a versão antiga suspendeu.",
            "A janela original é uma referência para diagnóstico, não anotação independente.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configuracoes.json"))
    parser.add_argument("--output", type=Path, default=Path("politica_validacao.json"))
    args = parser.parse_args()
    try:
        result = validate(args.log, load_config(args.config).state)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Erro: {exc}\n")
    print(
        f"Eventos fora de combate na janela: {result['baseline_noncombat_events']} → {result['v07_noncombat_events']}"
    )
