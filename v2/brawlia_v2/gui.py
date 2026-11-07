"""One-click performance baseline + non-blocking update checks."""

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox

from . import __version__
from .capture import MSSBlueStacksSource, SyntheticSource
from .pipeline import Pipeline
from .updates import check_release, download_verified
from .window import enable_dpi_awareness

BG = "#101827"
PANEL = "#192536"
WHITE = "#eff6ff"
DIM = "#b0bfd0"
CYAN = "#49ded0"


def report_directory():
    base = os.getenv("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "BrawlIA-V2" / "relatorios"


def write_report(stats):
    path = report_directory()
    path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("v2_fase1_%Y%m%d_%H%M%S.json")
    output = path / filename
    stats.update({"version": __version__, "phase": "capture_only"})
    output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"BrawlIA V2 · {__version__}")
        self.geometry("850x540")
        self.minsize(680, 470)
        self.configure(bg=BG)
        self.pipeline = None
        self.finalizing = False
        self.update_running = False
        self.remote_release = None
        self.is_frozen = getattr(sys, "frozen", False)
        self.status = tk.StringVar(value="Pronto. Abra o BlueStacks em modo janela.")
        self.update_status = tk.StringVar(value="Atualizações: verificando disponibilidade...")
        self.capture_label = tk.StringVar(value="—")
        self.analysis_label = tk.StringVar(value="—")
        self.latency_label = tk.StringVar(value="—")
        self.queue_label = tk.StringVar(value="—")
        self.synthetic = tk.BooleanVar(value=False)
        self._draw()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(250, self._tick)
        self.after(1200, self.check_update)

    def _label(self, parent, text, size=11, fg=WHITE, **kwargs):
        return tk.Label(parent, text=text, bg=parent["bg"], fg=fg,
                        font=("Segoe UI", size), **kwargs)

    def _draw(self):
        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True, padx=28, pady=20)
        self._label(root, "BrawlIA V2", 27).pack(anchor="w")
        self._label(root, "Fase 1 · Captura de quadros a 30 FPS · Sem OCR ou gravação de imagens",
                    10, DIM).pack(anchor="w", pady=(0, 18))
        commands = tk.Frame(root, bg=BG)
        commands.pack(fill="x", pady=(0, 12))
        self.run_button = tk.Button(commands, text="INICIAR TESTE 30 FPS", command=self.toggle,
                                    bg=CYAN, fg=BG, font=("Segoe UI Semibold", 12),
                                    relief="flat", padx=18, pady=12)
        self.run_button.pack(side="left", padx=(0, 10))
        tk.Button(commands, text="ABRIR RELATÓRIOS", command=self.open_reports,
                  bg=PANEL, fg=WHITE, relief="flat", padx=12, pady=12).pack(side="left")
        tk.Checkbutton(root, text="Fonte sintética (diagnóstico sem BlueStacks)",
                       variable=self.synthetic, bg=BG, fg=DIM, selectcolor=PANEL,
                       activebackground=BG, activeforeground=WHITE).pack(anchor="w")
        self._label(root, "CAPTURA / PROCESSAMENTO REAIS", 10, DIM).pack(anchor="w", pady=(15, 7))
        metrics = tk.Frame(root, bg=BG)
        metrics.pack(fill="x")
        for title, variable in (("Captura FPS", self.capture_label),
                                ("Análise FPS", self.analysis_label),
                                ("Captura p95 ms", self.latency_label),
                                ("Quadros substituídos", self.queue_label)):
            cell = tk.Frame(metrics, bg=PANEL, highlightthickness=1, highlightbackground="#29384b")
            cell.pack(side="left", fill="x", expand=True, padx=(0, 8))
            self._label(cell, title, 9, DIM).pack(anchor="w", padx=12, pady=(10, 0))
            tk.Label(cell, textvariable=variable, bg=PANEL, fg=WHITE,
                     font=("Segoe UI Semibold", 20)).pack(anchor="w", padx=12, pady=(0, 12))
        self._label(root, "ESTADO", 10, DIM).pack(anchor="w", pady=(17, 4))
        tk.Label(root, textvariable=self.status, bg=BG, fg=WHITE,
                 wraplength=740, anchor="w", justify="left").pack(fill="x")
        line = tk.Frame(root, bg="#334155", height=1)
        line.pack(fill="x", pady=16)
        updates = tk.Frame(root, bg=BG)
        updates.pack(fill="x")
        self.update_button = tk.Button(updates, text="VERIFICAR ATUALIZAÇÃO", command=self.check_update,
                                       bg=PANEL, fg=WHITE, relief="flat", padx=12, pady=8)
        self.update_button.pack(side="left", padx=(0, 14))
        tk.Label(updates, textvariable=self.update_status, bg=BG, fg=DIM,
                 wraplength=495, justify="left").pack(side="left", fill="x", expand=True)
        self._label(root, "Atualização verificada pelo GitHub Releases (SHA-256). "
                    "A instalação exige confirmação.", 9, DIM).pack(anchor="w", pady=(10, 0))

    def toggle(self):
        if self.finalizing:
            return
        if self.pipeline is None:
            source = SyntheticSource if self.synthetic.get() else MSSBlueStacksSource
            self.pipeline = Pipeline(source).start()
            self.synthetic_checkbox_state = self.synthetic.get()
            self.run_button.config(text="PARAR E SALVAR MÉTRICAS", bg="#f5a6a6")
            self.status.set("Capturando quadros novos, meta de 30 FPS.")
        else:
            self.finalizing = True
            self.run_button.config(state="disabled")
            pipeline = self.pipeline
            def finish():
                report = pipeline.stop()
                try:
                    filename = write_report(report)
                    self.after(0, lambda: self._finished(report, filename, None))
                except Exception as exc:
                    self.after(0, lambda exc=str(exc): self._finished(report, None, exc))
            threading.Thread(target=finish, daemon=True).start()

    def _finished(self, report, filename, error):
        self.pipeline = None
        self.finalizing = False
        self.run_button.config(text="INICIAR TESTE 30 FPS", state="normal", bg=CYAN)
        if report.get("error"):
            self.status.set("Erro na captura: " + report["error"])
        elif error:
            self.status.set("Não foi possível salvar métricas: " + error)
        else:
            self.status.set(f"Métricas salvas em {filename} · {report['capture_count']} quadros")

    def _tick(self):
        if self.pipeline is not None:
            report = self.pipeline.snapshot()
            self.capture_label.set(f"{report['capture_recent_fps']:.1f}")
            self.analysis_label.set(f"{report['process_recent_fps']:.1f}")
            self.latency_label.set(f"{report['capture_p95_ms']:.1f}")
            self.queue_label.set(str(report["replaced_before_processing"]))
            if report["error"] and not self.finalizing:
                self.toggle()
        self.after(250, self._tick)

    def open_reports(self):
        folder = report_directory()
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(folder))
        else:
            messagebox.showinfo("Relatórios", str(folder))

    def _background(self, job, success):
        def run():
            try:
                value = job()
            except Exception as exc:
                self.after(0, lambda message=str(exc): self._update_failed(message))
            else:
                self.after(0, lambda: success(value))
        threading.Thread(target=run, daemon=True).start()

    def _update_failed(self, error):
        self.update_running = False
        self.update_button.config(state="normal", text="VERIFICAR ATUALIZAÇÃO")
        self.update_status.set("Não foi possível consultar o GitHub: " + error[:130])

    def check_update(self):
        if self.update_running:
            return
        self.update_running = True
        self.update_button.config(state="disabled")
        self.update_status.set("Consultando GitHub Releases...")
        self._background(check_release, self._update_found)

    def _update_found(self, release):
        self.update_running = False
        self.update_button.config(state="normal", text="VERIFICAR ATUALIZAÇÃO")
        self.remote_release = release
        if release is None:
            self.update_status.set(f"Você possui a versão mais recente ({__version__}).")
            return
        self.update_status.set(f"Nova versão {release.version} disponível.")
        self.update_button.config(text="INSTALAR ATUALIZAÇÃO", command=self.install_update)

    def install_update(self):
        release = self.remote_release
        if release is None or self.update_running:
            return
        if not self.is_frozen:
            messagebox.showinfo("Modo de desenvolvimento", "OTA disponível no executável Windows, não no Python.")
            return
        if self.pipeline is not None:
            messagebox.showwarning("Captura ativa", "Pare a captura antes de atualizar.")
            return
        helper = Path(sys.executable).with_name("BrawlIA-V2-Updater.exe")
        if not helper.is_file():
            messagebox.showerror("Atualizador ausente", "Extraia os dois executáveis do ZIP do GitHub na mesma pasta.")
            return
        if not messagebox.askyesno("Atualizar BrawlIA V2", f"Instalar versão {release.version} e reiniciar?"):
            return
        self.update_running = True
        self.update_button.config(state="disabled")
        self.update_status.set("Baixando e verificando SHA-256...")
        folder = Path(sys.executable).resolve().parent / ".updates" / release.version
        self._background(lambda: download_verified(release, folder),
                         lambda result: self._launch_updater(helper, result))

    def _launch_updater(self, helper, result):
        staged, digest = result
        try:
            subprocess.Popen([
                str(helper), "--apply", "--pid", str(os.getpid()),
                "--target", str(Path(sys.executable).resolve()),
                "--staged", str(staged), "--sha256", digest,
            ], cwd=str(helper.parent), close_fds=True)
        except OSError as exc:
            self._update_failed(str(exc))
            return
        self.status.set("Atualização verificada. Reiniciando aplicativo...")
        self.after(150, self.destroy)

    def _close(self):
        if self.pipeline is not None:
            self.toggle()
            self.status.set("Finalizando o teste antes de fechar...")
            self.after(500, self._wait_close)
        else:
            self.destroy()

    def _wait_close(self):
        if self.finalizing:
            self.after(500, self._wait_close)
        else:
            self.destroy()


def main():
    enable_dpi_awareness()
    App().mainloop()


if __name__ == "__main__":
    main()
