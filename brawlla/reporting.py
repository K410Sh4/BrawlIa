"""Serialização e apresentação do relatório, separadas da agregação de métricas."""

import json
from pathlib import Path


def save_report(summary: dict, destination: Path) -> None:
    from html import escape

    destination = Path(destination)
    if destination.suffix.lower() != ".json":
        raise ValueError(
            "Escolha um destino com extensão .json; o HTML é criado ao lado."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    recording = summary.get("recording")
    recording_html = ""
    if recording:
        state = (
            "desligado"
            if not recording["video_enabled"]
            else "verificado" if recording["video_verified"] else "sem vídeo verificado"
        )
        rows = [
            ("Vídeo", state),
            ("Quadros de vídeo", recording["video_frames"]),
            ("Imagens brutas salvas", recording["images_saved"]),
            ("Pasta da sessão", recording["session_folder"]),
            (
                "Pasta das imagens",
                recording.get("samples_folder") or "Coleta desligada",
            ),
        ]
        recording_html = (
            "<h2>Arquivos salvos</h2><table>"
            + "".join(
                f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>"
                for k, v in rows
            )
            + "</table>"
        )
        for key in ("video_error", "samples_error"):
            if recording.get(key):
                recording_html += "<p>" + escape(recording[key]) + "</p>"
    coverage = summary["player_candidate_coverage_pct"]
    match_coverage = summary.get("inferred_match_candidate_coverage_pct")
    diagnostics = [
        (
            "Cobertura na janela inferida",
            "—" if match_coverage is None else f"{match_coverage:.1f}%",
        ),
        (
            "Leitura suspensa na janela",
            str(summary.get("analysis_suspended_in_match_s", 0)) + " s",
        ),
        ("Combate com HUD incerto", summary.get("uncertain_combat_frames", 0)),
        ("IDs de candidato a jogador", summary.get("unique_player_track_ids", 0)),
    ]
    diagnostics_html = "".join(
        f"<article><small>{escape(str(k))}</small><strong>{escape(str(v))}</strong></article>"
        for k, v in diagnostics
    )
    reasons = (
        "".join(
            f'<tr><td>{escape(k.replace("_"," "))}</td><td>{v}</td></tr>'
            for k, v in summary.get("track_event_counts", {}).items()
        )
        or '<tr><td colspan="2">Sem eventos detalhados neste registro.</td></tr>'
    )
    cells = [
        ("Análises", summary["analyzed_frames"]),
        ("Duração observada", f"{summary['observation_duration_s']:.1f} s"),
        ("Análises por segundo", summary["analysis_fps_over_observation"]),
        (
            "Cobertura do jogador em combate",
            "—" if coverage is None else f"{coverage:.1f}%",
        ),
        ("IDs temporários", summary["unique_track_ids"]),
        ("Mudanças de estado", summary["state_transitions"]),
    ]
    cards = "".join(
        f"<article><small>{escape(str(k))}</small><strong>{escape(str(v))}</strong></article>"
        for k, v in cells
    )
    labels = {
        "pre_partida": "Preparação",
        "combate": "Combate",
        "derrota": "Derrota",
        "desconhecido": "Desconhecido",
        "resultados": "Resultados",
    }
    stages = {
        "state": "Estado",
        "colors": "Cores e cenário",
        "classification": "Classificação de barras",
        "tracking": "Rastreamento",
        "overlay": "Marcações",
        "total": "Análise completa",
        "pipeline.capture": "Captura",
        "pipeline.resize": "Redimensionamento",
        "pipeline.analysis_with_resize": "Análise + redimensionamento",
        "pipeline.frame_age_at_result": "Idade do quadro no resultado",
    }
    timeline = "".join(
        f'<tr><td>{v["elapsed_s"]:.2f} s</td><td>{escape(labels.get(v["state"],v["state"]))}</td></tr>'
        for v in summary["timeline"]
    )
    methods = {
        "nome_verde_e_barra": "Nome e barra",
        "nome_verde_glifos_e_moldura": "Nome, números e moldura",
        "barra_verde_com_glifos": "Barra verde e números",
        "legado": "Registro de versão anterior",
    }
    evidence = "".join(
        f"<tr><td>{escape(methods.get(k,k))}</td><td>{v}</td></tr>"
        for k, v in summary.get("player_observations_by_evidence", {}).items()
    )
    if not evidence:
        evidence = (
            '<tr><td colspan="2">Nenhuma observação do jogador disponível.</td></tr>'
        )
    rejected = sum(
        summary.get(
            "health_bar_filter_rejections", summary.get("red_bar_filter_rejections", {})
        ).values()
    )
    rows = "".join(
        f'<tr><td>{escape(labels.get(k,k))}</td><td>{v}</td><td>{summary["state_duration_s"].get(k, 0):.2f} s</td></tr>'
        for k, v in summary["state_frame_counts"].items()
    )
    timings = "".join(
        f'<tr><td>{escape(stages.get(k,k))}</td><td>{v["mean"]:.2f} ms</td><td>{v["p95_recent"]:.2f} ms</td></tr>'
        for k, v in summary["timings_ms"].items()
    )
    if not timings:
        timings = '<tr><td colspan="3">Esta sessão não possui medições de tempo por etapa.</td></tr>'
    notes = "".join(f"<li>{escape(s)}</li>" for s in summary["limitations"])
    html = (
        """<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BrawlIA • Relatório de sessão</title><style>
    body{margin:0;background:#0b1016;color:#edf3fa;font:16px system-ui;line-height:1.6}main{max-width:1040px;margin:auto;padding:32px 20px}h1{margin-bottom:0}p,small{color:#a2b0c0} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}article{padding:20px;background:#151d27;border:1px solid #273647;border-radius:12px}strong{display:block;font-size:26px;color:#36cabb}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #273647}h2{font-size:20px;margin-top:30px}ul{padding-left:20px}header p{margin-top:4px}</style><main><header><h1>BrawlIA</h1><p>Relatório local de observação • detecções heurísticas</p></header>"""
        + f'<section class="grid">{cards}</section>{recording_html}<h2>Estados observados</h2><table><tr><th>Estado</th><th>Análises</th><th>Duração</th></tr>{rows}</table><h2>Tempo por etapa</h2><table><tr><th>Etapa</th><th>Média</th><th>p95 recente</th></tr>{timings}</table><h2>Leitura do jogador</h2><table><tr><th>Evidência visual</th><th>Observações</th></tr>{evidence}</table><p>Candidatos a barras de vida descartados pelos filtros: {rejected}. Essa contagem não confirma que todos eram falsos.</p><h2>Continuidade da sessão</h2><section class="grid">{diagnostics_html}</section><p>A janela é inferida dos estados e inclui interrupções; não é uma medida de acurácia. Combate incerto indica leitura mantida durante tolerância limitada de HUD.</p><details><summary>Motivos de criação, perda e reinício de IDs</summary><table><tr><th>Evento e motivo</th><th>Contagem</th></tr>{reasons}</table></details><h2>Linha do tempo</h2><details><summary>Ver mudanças de estado</summary><table><tr><th>Momento</th><th>Estado</th></tr>{timeline}</table></details><h2>Como interpretar</h2><ul>{notes}</ul></main></html>'
    )
    destination.with_suffix(".html").write_text(html, encoding="utf-8")
