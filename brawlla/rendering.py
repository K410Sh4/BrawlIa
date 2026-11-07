"""Desenho de overlays; nunca altera a imagem de entrada."""

import cv2
from .config import VERSION_SHORT


def render_terrain(frame, scene):
    image = frame.copy()
    for kind, contours in scene:
        cv2.drawContours(
            image, contours, -1, (255, 175, 45) if kind == "bush" else (45, 220, 245), 1
        )
    return image


def render_analysis(frame, data, scene):
    image = render_terrain(frame, scene)
    h, w = frame.shape[:2]
    state = data["game_state"]
    active = state["name"] == "combate"
    tracked = data["tracked_objects"]
    colors = {
        "jogador_candidato": (55, 220, 95),
        "caixa_provavel": (55, 205, 240),
        "personagem_possivel": (70, 95, 250),
        "indefinido": (180, 170, 160),
    }
    labels = {
        "jogador_candidato": "JOGADOR?",
        "caixa_provavel": "CAIXA?",
        "personagem_possivel": "PESSOA?",
        "indefinido": "ALVO?",
    }
    for obj in tracked:
        x, y, bw, bh = (obj[k] for k in ("x", "y", "width", "height"))
        color = colors.get(obj["kind"], (200, 200, 200))
        cv2.rectangle(image, (x - 2, y - 2), (x + bw + 2, y + bh + 2), color, 2)
        identity = obj.get("participant_id")
        stable_id = f"{identity} " if identity else ""
        label = (
            f"{stable_id}#{obj['track_id']} "
            f"{labels.get(obj['kind'], 'ALVO?')} {obj['confidence']:.2f}"
        )
        if obj.get("health_status") == "low_fill_candidate":
            label += " VIDA BAIXA?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.43
        (tw, th), _ = cv2.getTextSize(label, font, font_size, 1)
        lx = max(0, min(w - tw - 12, x + bw + 10))
        ly = max(th + 8, min(h - 8, y + 8))
        cv2.rectangle(
            image, (lx - 4, ly - th - 5), (lx + tw + 4, ly + 5), (24, 30, 35), -1
        )
        cv2.putText(
            image, label, (lx, ly), font, font_size, (245, 248, 251), 1, cv2.LINE_AA
        )
    cv2.rectangle(image, (max(0, w - 432), 0), (w, 32), (22, 25, 30), -1)
    text = f"BRAWLLA V{VERSION_SHORT} | {state['name'].upper()}{'?' if state.get('uncertain') else ''} {state['confidence']:.2f} | LEITURA"
    cv2.putText(
        image,
        text,
        (max(5, w - 423), 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (225, 232, 239),
        1,
        cv2.LINE_AA,
    )
    if not active:
        roster = data.get("match_roster", {})
        if roster.get("status") == "detected":
            text = (
                f"Apresentacao: {roster.get('confirmed_names', 0)}/10 "
                "nomes confirmados"
            )
        elif roster.get("status") == "ocr_unavailable":
            text = "OCR indisponivel; gravacao visual preservada"
        else:
            text = "Deteccao de alvos suspensa fora do combate"
        cv2.rectangle(image, (8, h - 34), (min(w - 1, 490), h - 5), (22, 25, 30), -1)
        cv2.putText(
            image,
            text,
            (14, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (225, 232, 239),
            1,
            cv2.LINE_AA,
        )
    return image
