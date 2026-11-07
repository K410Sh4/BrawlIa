"""BrawlIA V1 — painel de captura automática, uma ação e um único botão.

Captura pixels da janela visível do BlueStacks. Não pressiona teclas, não
movimenta personagens e não interage com o aplicativo observado.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
import time

import cv2
from PIL import Image, ImageTk

from brawlla.capture import AutoWindowCapture, enable_dpi_awareness
from brawlla.config import SessionOptions, VERSION_SHORT, load_config
from brawlla.session import SessionController
from brawlla.recordings import open_recording_folder, recording_folder
from brawlla.storage import application_directory
from brawlla.ui_mailbox import UiMailbox

BG = "#090f18"
PANEL = "#141e2b"
BORDER = "#273546"
MUTED = "#9cabbc"
TEXT = "#ecf4fb"
ACCENT = "#44d9c4"
BUTTON_TEXT = "#061c20"


class Metric(tk.Frame):
    def __init__(self, parent, caption):
        super().__init__(
            parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER
        )
        self.value = tk.StringVar(value="—")
        tk.Label(
            self,
            text=caption.upper(),
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill="x", padx=17, pady=(13, 2))
        tk.Label(
            self,
            textvariable=self.value,
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 22),
            anchor="w",
        ).pack(fill="x", padx=17, pady=(0, 11))


class Brawlla(tk.Tk):
    """UI de operação deliberadamente simples; sessões gerenciadas fora do Tk."""

    def __init__(self):
        super().__init__()
        self.title(f"BrawlIA V{VERSION_SHORT} — Captura inteligente")
        self.geometry("1150x820")
        self.minsize(820, 610)
        self.configure(bg=BG)
        self.messages = UiMailbox()
        self.session = SessionController(self._offer)
        self._closing = False
        self._last_preview_time = 0.0
        self._photo = None
        self.last_data = None
        self.last_annotated = None

        self.status = tk.StringVar(
            value="Pronto. Abra o BlueStacks e inicie a captura automática."
        )
        self.origin = tk.StringVar(value="Fonte: descoberta automática ao iniciar")
        self.destination = tk.StringVar(
            value=f"Gravações: {self.session.root / 'gravacoes'}"
        )
        self.detail = tk.StringVar(
            value="Somente observação · Vídeo e dados salvos automaticamente"
        )
        self.roster_info = tk.StringVar(
            value="Participantes: aguardando apresentação da partida"
        )
        self._build()
        self.after(120, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    @property
    def active(self):
        return self.session.active

    def _offer(self, kind, payload):
        self.messages.offer(kind, payload)

    def _build(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=27, pady=(22, 14))
        row = tk.Frame(header, bg=BG)
        row.pack(fill="x")
        tk.Label(
            row,
            text="BrawlIA",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI Semibold", 26),
        ).pack(side="left")
        tk.Label(
            row,
            text=f"  V{VERSION_SHORT}    •    VISÃO AUTOMÁTICA",
            bg=BG,
            fg=ACCENT,
            font=("Segoe UI", 10),
        ).pack(side="left", padx=12)
        tk.Label(
            header,
            text="Captura, análise e vídeo configurados em 30 FPS; monitoramento em tempo real.",
            bg=BG,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 11),
        ).pack(fill="x", pady=(5, 0))

        action = tk.Frame(
            self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1
        )
        action.pack(fill="x", padx=27, pady=(0, 12))
        self.start_btn = tk.Button(
            action,
            text="▶  INICIAR CAPTURA AUTOMÁTICA",
            command=self._toggle,
            bg=ACCENT,
            fg=BUTTON_TEXT,
            activebackground="#68eddb",
            activeforeground=BUTTON_TEXT,
            relief="flat",
            cursor="hand2",
            font=("Segoe UI Semibold", 13),
            padx=22,
            pady=15,
        )
        self.start_btn.pack(side="left", padx=(18, 8), pady=16)
        self.folder_btn = tk.Button(
            action,
            text="ABRIR GRAVAÇÃO",
            command=self._open_recording,
            bg="#283848",
            fg=TEXT,
            activebackground="#394f61",
            activeforeground=TEXT,
            relief="flat",
            cursor="hand2",
            font=("Segoe UI Semibold", 10),
            padx=13,
            pady=16,
        )
        self.folder_btn.pack(side="left", padx=(0, 12), pady=16)
        tk.Label(
            action,
            textvariable=self.status,
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 10),
            wraplength=390,
            justify="left",
            anchor="w",
        ).pack(side="left", fill="both", expand=True, padx=(6, 16))

        metrics = tk.Frame(self, bg=BG)
        metrics.pack(fill="x", padx=27, pady=(0, 12))
        self.capture_metric = Metric(metrics, "Captura real")
        self.analysis_metric = Metric(metrics, "Análise real")
        self.latency_metric = Metric(metrics, "p95 de análise")
        self.age_metric = Metric(metrics, "Idade do quadro")
        for widget in (
            self.capture_metric,
            self.analysis_metric,
            self.latency_metric,
            self.age_metric,
        ):
            widget.pack(side="left", fill="x", expand=True, padx=(0, 9))

        preview_panel = tk.Frame(
            self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1
        )
        preview_panel.pack(fill="both", expand=True, padx=27, pady=(0, 12))
        tk.Label(
            preview_panel,
            text="VISUALIZAÇÃO AO VIVO",
            bg=PANEL,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI Semibold", 10),
        ).pack(fill="x", padx=16, pady=(12, 4))
        self.preview = tk.Label(
            preview_panel,
            text="A imagem da janela do jogo aparecerá aqui.",
            bg="#0b131d",
            fg=MUTED,
            font=("Segoe UI", 13),
        )
        self.preview.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", padx=27, pady=(0, 19))
        tk.Label(
            footer,
            textvariable=self.roster_info,
            bg=BG,
            fg=ACCENT,
            anchor="w",
            font=("Segoe UI Semibold", 10),
        ).pack(fill="x", pady=(0, 5))
        for variable, color in (
            (self.origin, MUTED),
            (self.destination, TEXT),
            (self.detail, MUTED),
        ):
            tk.Label(
                footer,
                textvariable=variable,
                bg=BG,
                fg=color,
                anchor="w",
                font=("Segoe UI", 9),
            ).pack(fill="x", pady=1)

    def _open_recording(self):
        """Abre a sessão atual; após reiniciar, encontra a última gravação."""
        try:
            folder = recording_folder(self.session.root, self.session.folder)
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
            open_recording_folder(folder)
            self.destination.set(f"Pasta de gravação: {folder}")
        except (OSError, ValueError) as exc:
            self.status.set(f"Não foi possível abrir a gravação: {exc}")
            if not self._closing:
                messagebox.showerror("Pasta de gravação", str(exc))

    def _toggle(self):
        if self.session.finalizing or self._closing:
            return
        if self.active:
            self._stop()
        else:
            self._start()

    def _start(self):
        if self.session.state != "idle":
            return
        options = SessionOptions(
            auto_fps=False,
            capture_fps=30,
            analysis_fps=30,
            max_width=1280,
            record_samples=False,
            record_video=True,
            video_fps=30,
        )
        try:
            # Janela descoberta dentro do worker, no próprio thread MSS.
            self.session.start(
                AutoWindowCapture,
                options,
                load_config(application_directory() / "configuracoes.json"),
            )
        except Exception as exc:
            self.status.set(f"Não foi possível iniciar: {exc}")
            if not self._closing:
                messagebox.showerror("Falha na captura", str(exc))
            return
        self.start_btn.configure(text="■  PARAR E SALVAR", bg="#f19b91")
        self.status.set("Capturando em modo de 30 FPS fixos; a taxa efetiva depende do PC.")
        self.destination.set(f"Gravando em: {self.session.folder}")
        self.detail.set(
            "Vídeo 30 FPS e análise JSONL · PNG desativado · Janela visível"
        )

    def _stop(self):
        if self.session.stop():
            self.start_btn.configure(
                text="FINALIZANDO ARQUIVOS…", state="disabled", bg="#58677a"
            )
            self.status.set("Finalizando vídeo e registros da sessão…")

    def _preview_frame(self, frame):
        now = time.monotonic()
        if now - self._last_preview_time < 0.14:
            return
        self._last_preview_time = now
        self.last_annotated = frame
        height, width = frame.shape[:2]
        box_w = max(160, self.preview.winfo_width() - 12)
        box_h = max(120, self.preview.winfo_height() - 12)
        ratio = min(box_w / width, box_h / height, 1.0)
        preview = cv2.resize(
            frame,
            (max(1, round(width * ratio)), max(1, round(height * ratio))),
            interpolation=cv2.INTER_AREA,
        )
        self._photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        )
        self.preview.configure(image=self._photo, text="")

    def _poll(self):
        for kind, payload in self.messages.drain():
            if kind == "frame":
                frame, data = payload
                self.last_data = data
                roster = data.get("match_roster", {})
                if roster.get("status") == "detected":
                    local = roster.get("local_participant_id") or "pendente"
                    self.roster_info.set(
                        f"Participantes: {roster.get('confirmed_names', 0)}/10 "
                        f"nomes confirmados · Você: {local}"
                    )
                elif roster.get("status") == "ocr_unavailable":
                    self.roster_info.set(
                        "Participantes: OCR indisponível, consulte a documentação"
                    )
                if not self._closing:
                    self._preview_frame(frame)
                    source = data.get("capture", {}).get("source_label")
                    if source:
                        self.origin.set(f"Fonte: {source}")
            elif kind == "finalized":
                result, sample_error = payload
                if self._closing:
                    self.destroy()
                    return
                self.start_btn.configure(
                    text="▶  INICIAR CAPTURA AUTOMÁTICA",
                    state="normal",
                    bg=ACCENT,
                )
                if self.session.folder:
                    self.destination.set(f"Última gravação: {self.session.folder}")
                if result and result.get("error"):
                    self.status.set("Falha: " + result["error"])
                elif result and result.get("verified"):
                    self.status.set(
                        f"Gravação concluída: {result.get('written', 0)} quadros."
                    )
                else:
                    self.status.set("Sessão encerrada. Consulte os registros.")
                if sample_error:
                    self.detail.set("Falha em amostras: " + sample_error)
                elif result:
                    self.detail.set(
                        "Vídeo, histórico e autoavaliação salvos · Sem PNGs"
                    )
            elif kind == "recording_error":
                self.detail.set(
                    "Gravação: " + str(payload.get("error", "erro desconhecido"))
                )
            elif kind == "error":
                self.status.set("Erro: " + str(payload))

        pipe = self.session.pipeline
        if pipe is not None:
            stats = pipe.snapshot()
            self.capture_metric.value.set(f"{stats['capture_fps']:.1f} FPS")
            self.analysis_metric.value.set(f"{stats['analysis_fps']:.1f} FPS")
            self.latency_metric.value.set(f"{stats['analysis_p95_ms']:.0f} ms")
            self.age_metric.value.set(f"{stats['frame_age_ms']:.0f} ms")
            if self.active:
                self.status.set(
                    f"Meta fixa: 30 FPS · Captura real: {stats['capture_fps']:.1f}"
                    f" · Análise real: {stats['analysis_fps']:.1f} FPS"
                )
            if self.active and stats.get("error"):
                self.status.set("Erro de captura: " + stats["error"])
                self._stop()
            flags = stats.get("quality", {}).get("flags", [])
            if flags:
                self.detail.set("Atenção à visibilidade: " + ", ".join(flags))
        if self._closing and self.session.state == "idle":
            self.destroy()
            return
        self.after(120, self._poll)

    def _close(self):
        self._closing = True
        if self.session.active:
            self._stop()
        if self.session.finalizing:
            self.withdraw()
        else:
            self.destroy()


if __name__ == "__main__":
    enable_dpi_awareness()
    Brawlla().mainloop()
