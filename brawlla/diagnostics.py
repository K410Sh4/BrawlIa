"""Mensagens de diagnóstico distinguem processamento do quadro e leitura de alvos."""


def target_status(state):
    name = state.get("name", "desconhecido")
    if name == "combate":
        return {
            "enabled": True,
            "reason": "combat_uncertain" if state.get("uncertain") else "combat",
            "message": (
                "Alvos: ativos • HUD incerto"
                if state.get("uncertain")
                else "Alvos: ativos"
            ),
        }
    messages = {
        "pre_partida": "Alvos suspensos: menu reconhecido",
        "derrota": "Alvos suspensos: eliminação reconhecida",
        "resultados": "Alvos suspensos: tela de resultados",
        "desconhecido": "Alvos suspensos: HUD não reconhecido",
    }
    return {
        "enabled": False,
        "reason": name,
        "message": messages.get(name, "Alvos suspensos: estado não reconhecido"),
    }


def format_frame_counts(stats):
    captured, analyzed, replaced = (
        stats.get(k, stats.get("skipped", 0) if k == "replaced_before_analysis" else 0)
        for k in ("captured", "analyzed", "replaced_before_analysis")
    )
    return (
        f"Quadros capturados: {captured}  |  Analisados: {analyzed}"
        f"  |  Substituídos antes da análise: {replaced}"
    )


def format_processing_health(stats):
    if stats.get("error"):
        return "Erro na leitura: " + stats["error"]
    if stats.get("capture_stalled"):
        return "Sem novo quadro de captura há mais de 2 s; verifique a fonte."
    if stats.get("analysis_stalled"):
        return "A captura continua, mas a análise está sem resultados há mais de 2 s."
    if not stats.get("analyzed"):
        return "Aguardando o primeiro resultado de análise."
    return "Leitura de quadros funcionando."


def format_recording_status(snapshot):
    if snapshot.get("error"):
        return "Gravação com falha: " + str(snapshot["error"])
    if not snapshot.get("video_requested"):
        return "Vídeo desligado; serão salvos histórico e diagnóstico da sessão."
    video = snapshot.get("video") or {}
    if video.get("error"):
        return "Gravação com falha: " + str(video["error"])
    if video.get("limit_reached"):
        return "Gravação interrompida: " + video["limit_reached"]
    if video.get("done"):
        if video.get("verified") and video.get("written"):
            return f"Vídeo salvo e verificado: {video['written']} quadros."
        return "Nenhum quadro de vídeo salvo nesta sessão."
    if video.get("written"):
        return f"Gravando vídeo: {video['written']} quadros • {video.get('records',0)} eventos."
    return "Gravador preparado; aguardando o primeiro quadro de análise."
