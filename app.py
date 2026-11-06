"""Interface de observação do BrawlIA; ciclo de sessão delegado ao controlador."""

from __future__ import annotations

from pathlib import Path
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from brawlla.capture import (
    AdbCapture,
    WindowCapture,
    list_game_windows,
    enable_dpi_awareness,
)
from brawlla.dataset import crop_frame, export_dataset
from brawlla.review_ui import CalibrationWindow, ReviewWindow
from brawlla.storage import new_session_id
from brawlla.engine import analyze
from brawlla.session import SessionController
from brawlla.config import SessionOptions, load_config, VERSION_SHORT
from brawlla.storage import application_directory, load_preferences, save_preferences
from brawlla.ui_mailbox import UiMailbox
from brawlla.diagnostics import (
    target_status,
    format_recording_status,
    format_frame_counts,
    format_processing_health,
)
from brawlla.metrics import summarize_log
from brawlla.reporting import save_report

BG = "#0b1016"
PANEL = "#151d27"
TEXT = "#edf3fa"
MUTED = "#a2b0c0"
ACCENT = "#19b9ac"


class Brawlla(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Brawlla — Visão V{VERSION_SHORT} | Sessão contínua")
        self.geometry("1290x920")
        self.minsize(1120, 720)
        self.configure(bg=BG)
        self.photo = None
        self.last_annotated = None
        self.last_data = None
        self.session = SessionController(self._offer)
        self._closing = False
        self.windows = []
        self.messages = UiMailbox()
        self.report_pending = False
        self.export_pending = False
        self.pending_single = False
        preference_error = None
        try:
            preferences = load_preferences(self.session.root)
        except Exception as exc:
            preferences = SessionOptions(record_video=True)
            preference_error = str(exc)
        self.source_type = tk.StringVar(value="Janela (rápido)")
        self.window_name = tk.StringVar(value="")
        self.capture_fps = tk.StringVar(value=str(preferences.capture_fps))
        self.vision_fps = tk.StringVar(value=str(preferences.analysis_fps))
        self.analysis_width = tk.StringVar(value=str(preferences.max_width))
        self.port = tk.StringVar(value="5555")
        self.record_samples = tk.BooleanVar(value=preferences.record_samples)
        self.sample_fps = tk.StringVar(value=str(preferences.sample_fps))
        self.record_video = tk.BooleanVar(value=preferences.record_video)
        self.video_fps = tk.StringVar(value=str(preferences.video_fps))
        self.roi, self.source_aspect = preferences.roi, preferences.source_aspect
        self.burst_fps = tk.StringVar(value=str(preferences.burst_fps))
        self.roi_status = tk.StringVar(
            value="Recorte calibrado" if self.roi else "Área completa"
        )
        self.status = tk.StringVar(
            value="Pronto — análise de imagem, sem controlar o jogo."
        )
        self.health = tk.StringVar(
            value="Tempos: captura —  |  análise p95 —  |  idade do quadro —"
        )
        self.metrics = tk.StringVar(value="FPS captura: —  |  FPS análise: —")
        self.frame_counts = tk.StringVar(
            value="Quadros capturados: 0  |  Analisados: 0  |  Substituídos antes da análise: 0"
        )
        self.processing_status = tk.StringVar(value="Aguardando início da leitura.")
        self.recording_status = tk.StringVar(
            value=(
                "Vídeo ativado para a próxima leitura."
                if preferences.record_video
                else "Vídeo desligado: somente leitura."
            )
        )
        self.save_destination = tk.StringVar(
            value="Destino: " + str(self.session.root / "gravacoes")
        )
        self.detections_text = tk.StringVar(
            value="Jogador: —  |  Barras vermelhas: —  |  Arbustos: —  |  Paredes: —"
        )
        self._build()
        self._refresh_windows()
        if preference_error:
            self.status.set(
                "Preferências inválidas; usando padrões. " + preference_error
            )
        self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    @property
    def pipeline(self):
        return self.session.pipeline

    @property
    def recorder(self):
        return self.session.sampler

    @property
    def video_recorder(self):
        return self.session.video

    @property
    def active(self):
        return self.session.active

    @property
    def _finalizing(self):
        return self.session.finalizing

    def _label(self, parent, title: str) -> None:
        tk.Label(parent, text=title, fg=MUTED, bg=PANEL, font=("Segoe UI", 9)).pack(
            side="left", padx=(11, 4)
        )

    def _combo(self, parent, variable, values, width=10):
        return ttk.Combobox(
            parent, textvariable=variable, values=values, state="readonly", width=width
        )

    def _button(self, parent, title: str, action):
        return tk.Button(
            parent,
            text=title,
            command=action,
            bg="#273647",
            fg=TEXT,
            activebackground="#375367",
            activeforeground=TEXT,
            relief="flat",
            font=("Segoe UI", 10),
            padx=10,
            pady=8,
            cursor="hand2",
        )

    def _build(self) -> None:
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=20, pady=(16, 8))
        tk.Label(
            header, text="BRAWLLA", fg=TEXT, bg=BG, font=("Segoe UI", 20, "bold")
        ).pack(side="left")
        tk.Label(
            header,
            text=f"V{VERSION_SHORT}  |  CAPTURA SINCRONIZADA + REVISÃO",
            fg=ACCENT,
            bg=BG,
            font=("Segoe UI", 10),
        ).pack(side="left", padx=22)

        settings = tk.Frame(self, bg=PANEL)
        settings.pack(fill="x", padx=20, pady=5, ipady=9)
        self._label(settings, "Fonte")
        self.source_combo = self._combo(
            settings, self.source_type, ["Janela (rápido)", "ADB (compatível)"], 17
        )
        self.source_combo.pack(side="left")
        self.source_combo.bind("<<ComboboxSelected>>", lambda _: self._source_changed())
        self._label(settings, "Janela")
        self.window_combo = self._combo(settings, self.window_name, [], 32)
        self.window_combo.pack(side="left")
        self.refresh_btn = self._button(
            settings, "Atualizar janelas", self._refresh_windows
        )
        self.refresh_btn.pack(side="left", padx=6)
        self._label(settings, "ADB")
        self.port_entry = tk.Entry(
            settings,
            width=6,
            textvariable=self.port,
            bg=BG,
            fg=TEXT,
            relief="flat",
            insertbackground=TEXT,
        )
        self.port_entry.pack(side="left")

        speed = tk.Frame(self, bg=PANEL)
        speed.pack(fill="x", padx=20, pady=5, ipady=9)
        self._label(speed, "Captura FPS")
        self.capture_combo = self._combo(
            speed, self.capture_fps, ["5", "10", "15", "30", "60"], 5
        )
        self.capture_combo.pack(side="left")
        self._label(speed, "Análise máx. FPS")
        self.vision_combo = self._combo(
            speed, self.vision_fps, ["5", "10", "15", "30", "60"], 5
        )
        self.vision_combo.pack(side="left")
        self._label(speed, "Largura da análise")
        self.width_combo = self._combo(
            speed, self.analysis_width, ["640", "960", "1280", "1600", "1920"], 6
        )
        self.width_combo.pack(side="left")
        self.samples_check = tk.Checkbutton(
            speed,
            text="Coletar frames brutos",
            command=self._save_recording_choice,
            variable=self.record_samples,
            bg=PANEL,
            fg=TEXT,
            selectcolor=BG,
            activebackground=PANEL,
            activeforeground=TEXT,
        )
        self.samples_check.pack(side="left", padx=(14, 0))
        self._label(speed, "Salvar FPS")
        self.sample_combo = self._combo(
            speed, self.sample_fps, ["1", "2", "5", "10", "15"], 5
        )
        self.sample_combo.pack(side="left")

        recording = tk.Frame(self, bg=PANEL)
        recording.pack(fill="x", padx=20, pady=5, ipady=9)
        self.video_check = tk.Checkbutton(
            recording,
            text="Gravar vídeo da análise (com marcações)",
            command=self._save_recording_choice,
            variable=self.record_video,
            bg=PANEL,
            fg=TEXT,
            selectcolor=BG,
            activebackground=PANEL,
            activeforeground=TEXT,
        )
        self.video_check.pack(side="left", padx=(13, 7))
        self._label(recording, "FPS do vídeo")
        self.video_combo = self._combo(
            recording, self.video_fps, ["5", "10", "15", "30", "60"], 5
        )
        self.video_combo.pack(side="left")
        tk.Label(
            recording,
            text="Vídeo + JSONL  |  destino exibido abaixo  |  limite: 30 minutos",
            fg=MUTED,
            bg=PANEL,
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(18, 7))
        self._button(recording, "Abrir gravações", self._open_recordings).pack(
            side="right", padx=12
        )

        dataset_row = tk.Frame(self, bg=PANEL)
        dataset_row.pack(fill="x", padx=20, pady=4, ipady=5)
        self.calibrate_btn = self._button(
            dataset_row, "Calibrar recorte", self._calibrate
        )
        self.calibrate_btn.pack(side="left", padx=10)
        self.clear_roi_btn = self._button(
            dataset_row, "Área completa", lambda: self._apply_roi(None, 0)
        )
        self.clear_roi_btn.pack(side="left")
        tk.Label(dataset_row, textvariable=self.roi_status, bg=PANEL, fg=MUTED).pack(
            side="left", padx=10
        )
        self._label(dataset_row, "Rajada de dados (FPS)")
        self.burst_combo = self._combo(dataset_row, self.burst_fps, ["5", "10"], 5)
        self.burst_combo.pack(side="left")
        self._button(dataset_row, "Revisar amostras", self._review_samples).pack(
            side="right", padx=10
        )
        self._button(dataset_row, "Exportar dataset", self._export_dataset).pack(
            side="right"
        )

        actions = tk.Frame(self, bg=PANEL)
        actions.pack(fill="x", padx=20, pady=5, ipady=7)
        self.open_btn = self._button(actions, "Abrir imagem", self._open)
        self.open_btn.pack(side="left", padx=(12, 4))
        self.report_btn = self._button(
            actions, "Relatório de sessão", self._report_session
        )
        self.report_btn.pack(side="left", padx=4)
        self.single_btn = self._button(actions, "Captura única", self._single)
        self.single_btn.pack(side="left", padx=4)
        self.start_btn = self._button(actions, "Iniciar leitura", self._toggle)
        self.start_btn.pack(side="left", padx=4)
        self._button(actions, "Salvar análise", self._save).pack(side="right", padx=12)

        self.preview = tk.Label(
            self,
            text="Escolha a janela do BlueStacks e clique em Iniciar leitura.",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 12),
        )
        self.preview.pack(fill="both", expand=True, padx=20, pady=(7, 7))
        self.preview.bind("<Configure>", lambda _: self._render())
        tk.Label(
            self,
            textvariable=self.status,
            bg=BG,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 10),
        ).pack(fill="x", padx=20, pady=(2, 0))
        tk.Label(
            self,
            textvariable=self.metrics,
            bg=BG,
            fg=ACCENT,
            anchor="w",
            font=("Segoe UI", 11, "bold"),
        ).pack(fill="x", padx=20, pady=(4, 0))
        tk.Label(
            self,
            textvariable=self.frame_counts,
            bg=BG,
            fg=TEXT,
            anchor="w",
            font=("Segoe UI", 10),
        ).pack(fill="x", padx=20, pady=(3, 0))
        tk.Label(
            self,
            textvariable=self.processing_status,
            bg=BG,
            fg=ACCENT,
            anchor="w",
            font=("Segoe UI", 10),
            wraplength=1080,
            justify="left",
        ).pack(fill="x", padx=20, pady=(3, 0))
        tk.Label(
            self,
            textvariable=self.recording_status,
            bg=BG,
            fg=TEXT,
            anchor="w",
            font=("Segoe UI", 10),
            wraplength=1080,
            justify="left",
        ).pack(fill="x", padx=20, pady=(3, 0))
        tk.Label(
            self,
            textvariable=self.save_destination,
            bg=BG,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 9),
            wraplength=1080,
            justify="left",
        ).pack(fill="x", padx=20, pady=(3, 0))
        tk.Label(
            self,
            textvariable=self.health,
            bg=BG,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 10),
        ).pack(fill="x", padx=20, pady=(3, 0))
        tk.Label(
            self,
            textvariable=self.detections_text,
            bg=BG,
            fg=TEXT,
            anchor="w",
            font=("Segoe UI", 10),
        ).pack(fill="x", padx=20, pady=(4, 14))
        self._source_changed()

    def _source_changed(self) -> None:
        use_adb = self.source_type.get().startswith("ADB")
        self.port_entry.configure(state="normal" if use_adb else "disabled")
        self.window_combo.configure(state="disabled" if use_adb else "readonly")
        self.refresh_btn.configure(state="disabled" if use_adb else "normal")

    def _refresh_windows(self) -> None:
        self.windows = list_game_windows()
        names = [
            f"{w.title} [{w.width}×{w.height}] (#{w.handle})" for w in self.windows
        ]
        self.window_combo.configure(values=names)
        if names:
            self.window_combo.current(0)
            self.status.set(
                f"{len(names)} janela(s) encontrada(s). Mantenha o jogo visível."
            )
        else:
            self.window_name.set("")
            self.status.set(
                "Nenhuma janela BlueStacks detectada. Abra o emulador ou use ADB."
            )

    def _source_factory(self):
        if self.source_type.get().startswith("ADB"):
            port = int(self.port.get())
            return lambda: AdbCapture(port)
        idx = self.window_combo.current()
        if idx < 0 or idx >= len(self.windows):
            raise ValueError(
                "Selecione uma janela BlueStacks. Clique em Atualizar janelas."
            )
        handle = self.windows[idx].handle
        return lambda: WindowCapture(handle)

    def _offer(self, kind: str, data) -> None:
        self.messages.offer(kind, data)

    def _set_running_controls(self, running):
        state = "disabled" if running else "readonly"
        for combo in (
            self.source_combo,
            self.capture_combo,
            self.vision_combo,
            self.width_combo,
            self.sample_combo,
            self.burst_combo,
            self.video_combo,
        ):
            combo.configure(state=state)
        for button in (self.single_btn, self.open_btn):
            button.configure(state="disabled" if running else "normal")
        self.window_combo.configure(state="disabled" if running else "readonly")
        self.refresh_btn.configure(state="disabled" if running else "normal")
        self.port_entry.configure(state="disabled" if running else "normal")
        for toggle in (
            self.samples_check,
            self.video_check,
            self.calibrate_btn,
            self.clear_roi_btn,
        ):
            toggle.configure(state="disabled" if running else "normal")
        if not running:
            self._source_changed()

    def _report_session(self):
        if self.report_pending:
            return
        source = filedialog.askopenfilename(
            title="Escolha analise.jsonl de uma sessão",
            filetypes=[("Registro de sessão", "*.jsonl")],
        )
        if not source:
            return
        destination = filedialog.asksaveasfilename(
            title="Salvar relatório JSON + HTML",
            initialdir=str(Path(source).parent),
            initialfile="relatorio.json",
            defaultextension=".json",
            filetypes=[("Relatório JSON", "*.json")],
        )
        if not destination:
            return
        if Path(destination).resolve() == Path(source).resolve():
            messagebox.showerror(
                "Destino inválido",
                "Escolha outro arquivo para preservar o registro original.",
            )
            return
        html_path = Path(destination).with_suffix(".html")
        if html_path.exists() and not messagebox.askyesno(
            "Substituir relatório",
            f"O relatório HTML já existe. Substituir {html_path.name}?",
        ):
            return
        self.report_pending = True
        self.report_btn.configure(state="disabled")
        self.status.set("Analisando registro de sessão...")

        def work():
            try:
                save_report(summarize_log(Path(source)), Path(destination))
                self._offer("report_done", (str(html_path), None))
            except Exception as exc:
                self._offer("report_done", (None, str(exc)))

        threading.Thread(target=work, name="BrawllaReport", daemon=False).start()

    def _open_recordings(self) -> None:
        folder = self.session.folder or self.session.root / "gravacoes"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(str(folder))
            else:
                self.status.set("Pasta de gravações: " + str(folder))
        except OSError as exc:
            messagebox.showerror("Pasta indisponível", str(exc))

    def _session_options(self):
        return SessionOptions(
            capture_fps=int(self.capture_fps.get()),
            analysis_fps=int(self.vision_fps.get()),
            max_width=int(self.analysis_width.get()),
            record_samples=self.record_samples.get(),
            sample_fps=int(self.sample_fps.get()),
            record_video=self.record_video.get(),
            video_fps=int(self.video_fps.get()),
            roi=self.roi,
            source_aspect=self.source_aspect,
            burst_fps=int(self.burst_fps.get()),
        )

    def _save_recording_choice(self):
        try:
            save_preferences(self.session.root, self._session_options())
            self.recording_status.set(
                "Vídeo ativado para a próxima leitura."
                if self.record_video.get()
                else "Vídeo desligado: somente leitura."
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Não foi possível salvar a preferência", str(exc))

    def _toggle(self) -> None:
        if self._finalizing:
            return
        if self.active:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        if self._finalizing or self.pending_single or self.export_pending:
            return
        try:
            self.messages.clear_preview()
            self.frame_counts.set(
                "Quadros capturados: 0  |  Analisados: 0  |  Substituídos antes da análise: 0"
            )
            self.processing_status.set("Aguardando o primeiro resultado.")
            self.last_data = None
            options = self._session_options()
            source = self._source_factory()
            save_preferences(self.session.root, options)
            self.session.start(
                source,
                options,
                load_config(application_directory() / "configuracoes.json"),
            )
            self.save_destination.set("Sessão: " + str(self.session.folder))
            self.recording_status.set(
                format_recording_status(self.session.recording_snapshot())
            )
            self.start_btn.configure(text="Parar leitura e finalizar gravação")
            self._set_running_controls(True)
            self.status.set(
                "Leitura iniciada"
                + (
                    " | vídeo solicitado; aguardando primeiro quadro"
                    if options.record_video
                    else " | vídeo desligado"
                )
                + "."
            )
        except Exception as exc:
            if self._finalizing:
                self.start_btn.configure(
                    text="Finalizando gravação...", state="disabled"
                )
            messagebox.showerror("Não foi possível iniciar", str(exc))

    def _stop(self) -> None:
        if self.session.stop():
            self._set_running_controls(True)
            self.start_btn.configure(text="Finalizando gravação...", state="disabled")
            self.status.set("Parando captura e finalizando os arquivos. Aguarde.")

    def _apply_roi(self, roi, aspect):
        if self.active or self._finalizing:
            return
        self.roi, self.source_aspect = roi, aspect
        self.roi_status.set("Recorte calibrado" if roi else "Área completa")
        save_preferences(self.session.root, self._session_options())

    def _calibrate(self):
        if self.pending_single or self.active or self._finalizing:
            return
        try:
            factory = self._source_factory()
        except Exception as exc:
            messagebox.showwarning("Fonte indisponível", str(exc))
            return
        self.pending_single = True
        self.start_btn.configure(state="disabled")

        def work():
            source = None
            try:
                source = factory()
                self._offer("calibration", source.grab())
            except Exception as exc:
                self._offer("error", str(exc))
            finally:
                if source:
                    source.close()

        threading.Thread(target=work, daemon=True).start()

    def _review_samples(self):
        if self.active or self._finalizing:
            messagebox.showinfo(
                "Aguarde", "Pare a leitura para revisar os arquivos finalizados."
            )
            return
        folder = filedialog.askdirectory(
            title="Escolha a sessão com manifest.jsonl",
            initialdir=str(self.session.root / "capturas"),
        )
        if folder:
            try:
                ReviewWindow(self, folder)
            except Exception as exc:
                messagebox.showerror("Revisão indisponível", str(exc))

    def _export_dataset(self):
        if self.active or self._finalizing:
            messagebox.showinfo("Aguarde", "Pare a leitura antes de exportar.")
            return
        if self.export_pending:
            return
        self.export_pending = True
        root = self.session.root / "capturas"
        destination = root / "exportacoes" / new_session_id()
        self.status.set("Exportando revisões aprovadas...")

        def work():
            try:
                result = export_dataset(root, destination)
                self._offer("dataset_export", (str(destination), result, None))
            except Exception as exc:
                self._offer("dataset_export", (str(destination), None, str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _single(self) -> None:
        if self.pending_single or self.active or self._finalizing:
            return
        try:
            source_factory = self._source_factory()
        except Exception as exc:
            messagebox.showwarning("Fonte indisponível", str(exc))
            return
        self.pending_single = True
        self.start_btn.configure(state="disabled")
        self.single_btn.configure(state="disabled")
        self.status.set("Capturando uma imagem...")
        roi, aspect = self.roi, self.source_aspect

        def work():
            src = None
            result = None
            error = None
            try:
                src = source_factory()
                raw, geometry = crop_frame(src.grab(), roi, aspect)
                image, data = analyze(raw)
                data["capture"] = geometry
                result = (image, data)
            except Exception as exc:
                error = str(exc)
            finally:
                if src is not None:
                    try:
                        src.close()
                    except Exception as exc:
                        error = error or str(exc)
            self._offer("error", error) if error else self._offer("single", result)

        threading.Thread(target=work, daemon=True).start()

    def _open(self) -> None:
        if self.active or self._finalizing:
            return
        name = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.webp"), ("Todos", "*.*")]
        )
        if not name:
            return
        try:
            frame = cv2.imdecode(np.fromfile(name, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("Formato de imagem inválido")
            annotated, data = analyze(frame)
            self._display(annotated, data)
        except Exception as exc:
            messagebox.showerror("Falha ao carregar", str(exc))

    def _display(self, annotated: np.ndarray, data: dict) -> None:
        self.last_annotated, self.last_data = annotated, data
        counts = data.get("objects_count_by_kind", {})
        state = data.get("game_state", {})
        self.detections_text.set(
            f"Estado: {state.get('name', 'indefinido')} ({state.get('confidence', 0):.2f})"
            f"{' | HUD incerto' if state.get('uncertain') else ' | confirmando mudança' if state.get('held') else ''}"
            f"  |  Jogador?: {counts.get('jogador_candidato', 0)}"
            f"  |  Caixa?: {counts.get('caixa_provavel', 0)}"
            f"  |  Pessoa?: {counts.get('personagem_possivel', 0)}"
            f"  |  Indefinidos: {counts.get('indefinido', 0)}"
            f"  |  IDs: {len(data.get('tracked_objects', []))}"
        )
        self._render()

    def _render(self) -> None:
        if self.last_annotated is None:
            return
        w, h = max(1, self.preview.winfo_width() - 18), max(
            1, self.preview.winfo_height() - 18
        )
        img_h, img_w = self.last_annotated.shape[:2]
        scale = min(w / img_w, h / img_h, 1.0)
        new_w, new_h = max(1, round(img_w * scale)), max(1, round(img_h * scale))
        # OpenCV.INTER_AREA é adequado para preview e mais leve que LANCZOS a cada quadro.
        resized = cv2.resize(
            self.last_annotated, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        self.photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        )
        self.preview.configure(image=self.photo, text="")

    def _poll(self) -> None:
        # Um preview recente, todos os eventos de controle preservados.
        for kind, payload in self.messages.drain():
            if kind == "calibration":
                self.pending_single = False
                self.start_btn.configure(state="normal")
                if not self._closing:
                    CalibrationWindow(self, payload, self._apply_roi)
            elif kind in ("frame", "single"):
                annotated, data = payload
                self._display(annotated, data)
                if kind == "single":
                    self.pending_single = False
                    self.start_btn.configure(state="normal")
                    self.single_btn.configure(state="normal")
            elif kind == "finalized":
                result, sample_error = payload
                self.processing_status.set("Leitura encerrada.")
                self._set_running_controls(False)
                self.start_btn.configure(text="Iniciar leitura", state="normal")
                if result:
                    self.save_destination.set("Sessão: " + result.get("folder", ""))
                    if result.get("error"):
                        self.recording_status.set(
                            "Gravação com falha: " + result["error"]
                        )
                    elif result.get("video_disabled"):
                        self.recording_status.set(
                            "Vídeo desligado; histórico e diagnóstico salvos. PNGs: "
                            + str((result.get("samples") or {}).get("saved", 0))
                        )
                    elif result.get("written") and result.get("verified"):
                        self.recording_status.set(
                            "Vídeo salvo e verificado: "
                            + str(result["written"])
                            + " quadros. PNGs: "
                            + str((result.get("samples") or {}).get("saved", 0))
                        )
                    else:
                        self.recording_status.set(
                            "Nenhum quadro de vídeo salvo; consulte sessao.json."
                        )
                if result and result.get("error"):
                    self.status.set("Falha na sessão: " + result["error"])
                elif result and result.get("video_disabled"):
                    self.status.set(
                        "Sessão encerrada sem vídeo; histórico e diagnóstico salvos em "
                        + result["folder"]
                    )
                elif result and result.get("written", 0):
                    self.status.set(
                        f"Gravação finalizada: {result['written']} frames | relatório HTML em {result['folder']}"
                    )
                elif result:
                    self.status.set(
                        "Sem frames gravados. Pasta de sessão: " + result["folder"]
                    )
                else:
                    self.status.set(
                        "Leitura finalizada; nenhuma gravação de vídeo solicitada."
                    )
                if sample_error:
                    self.status.set("Erro ao salvar frames brutos: " + sample_error)
            elif kind == "recording_error":
                self.recording_status.set(
                    "Gravação com falha: " + str(payload.get("error"))
                )
            elif kind == "error":
                self.pending_single = False
                self.start_btn.configure(state="normal")
                self.single_btn.configure(state="normal")
                self.status.set("Erro: " + str(payload))

            elif kind == "dataset_export":
                self.export_pending = False
                path, result, error = payload
                self.status.set(
                    "Erro na exportação: " + error
                    if error
                    else "Dataset exportado: " + path
                )
                if not self._closing:
                    if error:
                        messagebox.showerror("Falha na exportação", error)
                    else:
                        messagebox.showinfo(
                            "Exportação concluída",
                            f"{result['images']} imagens revisadas; {result['annotations']} caixas.\n{path}\nOs JSONs referenciam os PNGs da pasta capturas.\nPoucas sessões podem deixar conjuntos vazios. Sessões antigas ignoradas: {len(result['legacy_sessions_skipped'])}.",
                        )
            elif kind == "report_done":
                self.report_pending = False
                self.report_btn.configure(state="normal")
                path, error = payload
                self.status.set(
                    "Erro no relatório: " + error
                    if error
                    else "Relatório salvo: " + path
                )
                if not error and not self._closing and os.name == "nt":
                    try:
                        os.startfile(path)
                    except OSError:
                        self.status.set("Relatório salvo; abra manualmente: " + path)

        pipe, sampler, video = self.pipeline, self.recorder, self.video_recorder
        if pipe is not None:
            s = pipe.snapshot()
            self.recording_status.set(
                format_recording_status(self.session.recording_snapshot())
            )
            save_txt = ""
            if sampler is not None:
                save_txt = f"  |  PNGs: {sampler.saved}/{sampler.max_samples} | inválidas: {sampler.invalid} | fila substituída: {sampler.skipped}"
                if sampler.limit_reached:
                    self.status.set(
                        "Limite de gravação atingido. Captura continua sem salvar novas imagens."
                    )
                elif sampler.error:
                    self.status.set("Erro de gravação: " + sampler.error)
            if video is not None:
                recorded = video.snapshot()
                save_txt += f"  |  Vídeo: {recorded['written']} frames ({recorded['dropped']} descartados)"
                if recorded["error"]:
                    self.status.set("Falha na gravação: " + recorded["error"])
                elif recorded["limit_reached"]:
                    self.status.set(
                        "Gravação interrompida: " + recorded["limit_reached"]
                    )
            self.health.set(
                f"Captura: {s['capture_ms']:.0f} ms  |  Análise p95 (120 quadros): {s['analysis_p95_ms']:.0f} ms"
                f"  |  Idade do quadro ao analisar: {s['frame_age_ms']:.0f} ms"
            )
            self.metrics.set(
                f"FPS captura: {s['capture_fps']:.1f}/{pipe.target_fps}"
                f"  |  FPS análise: {s['analysis_fps']:.1f}/{pipe.analysis_fps}"
                f"  |  Latência da análise: {s['last_latency_ms']:.0f} ms" + save_txt
            )
            self.frame_counts.set(format_frame_counts(s))
            message = format_processing_health(s)
            if (
                s["analyzed"]
                and self.last_data is not None
                and not s["analysis_stalled"]
                and not s["capture_stalled"]
                and not s["error"]
            ):
                status = self.last_data.get("target_analysis") or target_status(
                    self.last_data.get("game_state", {})
                )
                message += "  |  " + status["message"]
            quality_flags = s.get("quality", {}).get("flags", [])
            if quality_flags:
                message += " | Captura suspeita: " + ", ".join(quality_flags)
            self.processing_status.set(message)
            if self.active and s["error"]:
                self._stop()
                self.status.set("Erro na leitura: " + s["error"])
        if (
            self._closing
            and not self._finalizing
            and not self.pending_single
            and not self.report_pending
            and not self.export_pending
        ):
            self.destroy()
            return
        self.after(100, self._poll)

    def _save(self) -> None:
        if self.last_annotated is None:
            messagebox.showinfo("Sem análise", "Primeiro capture ou abra uma imagem.")
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile="brawlla-analise.png",
        )
        if not dest:
            return
        json_path = Path(dest).with_suffix(".json")
        if json_path.exists() and not messagebox.askyesno(
            "Substituir análise", f"O arquivo {json_path.name} já existe. Substituir?"
        ):
            return
        import json

        ok, image = cv2.imencode(".png", self.last_annotated)
        if not ok:
            messagebox.showerror("Erro", "Falha ao salvar PNG")
            return
        try:
            image.tofile(dest)
            json_path.write_text(
                json.dumps(self.last_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            messagebox.showerror("Falha ao salvar", str(exc))
            return
        self.status.set("Análise salva em " + str(Path(dest).parent))

    def _close(self) -> None:
        if (
            self.active
            or self._finalizing
            or self.pending_single
            or self.report_pending
            or self.export_pending
        ):
            self._closing = True
            self.withdraw()
            if self.active:
                self._stop()
        else:
            self.destroy()


if __name__ == "__main__":
    enable_dpi_awareness()
    Brawlla().mainloop()
