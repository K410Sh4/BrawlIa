"""Ciclo de sessão, gravação e diagnóstico persistente, independente do Tk."""

import json
from dataclasses import asdict
from pathlib import Path
import threading
import time
from .config import AnalyzerConfig, SessionOptions, VERSION
from .engine import SessionAnalyzer
from .stream import VisionPipeline
from .samples import SampleRecorder
from .observations import ObservationRecorder
from .provenance import provenance_snapshot
from .recording import AnalysisRecorder
from .reporting import save_report
from .storage import (
    user_data_directory,
    probe_directory,
    new_session_id,
    write_json,
    utc_now,
)


class SessionController:
    def __init__(self, notify, *, root=None):
        self.notify = notify
        self.root = (
            Path(root) if root is not None else user_data_directory()
        ).resolve()
        self.pipeline = self.sampler = self.video = self.observations = None
        self.state = "idle"
        self.folder = None
        self.options = None
        self.analyzer_config = None
        self.provenance = None
        self.started_utc = None
        self.last_summary = None
        self.recording_error = None
        self._lock = threading.RLock()
        self._audit_lock = threading.Lock()
        self._finalizer = None
        self._initial_error = None
        self._next_audit = 0
        self._notified_error = None

    @property
    def active(self):
        return self.state in ("starting", "running")

    @property
    def finalizing(self):
        return self.state == "stopping"

    def recording_snapshot(self):
        video_recorder, sample_recorder = self.video, self.sampler
        video = video_recorder.snapshot() if video_recorder else None
        samples = sample_recorder.snapshot() if sample_recorder else None
        if self.state == "idle" and self.last_summary:
            video = self.last_summary.get("video")
            samples = self.last_summary.get("samples")
        history = (
            self.observations.snapshot()
            if self.observations
            else (self.last_summary or {}).get("history")
        )
        return {
            "history": history,
            "video_requested": bool(self.options and self.options.record_video),
            "samples_requested": bool(self.options and self.options.record_samples),
            "folder": str(self.folder) if self.folder else "",
            "video": video,
            "samples": samples,
            "error": self.recording_error
            or (video or {}).get("error")
            or (samples or {}).get("error")
            or (history or {}).get("error"),
            "finalized": self.state == "idle" and self.options is not None,
        }

    def _summary(self, pipe=None, video=None, samples=None, *, final=False, error=None):
        return {
            "app_version": VERSION,
            "session_id": self.folder.name if self.folder else None,
            "read_only": True,
            "started_utc": self.started_utc,
            "updated_utc": utc_now(),
            "status": "finalizada" if final else self.state,
            "options": asdict(self.options),
            "analyzer_config": asdict(self.analyzer_config),
            "provenance": self.provenance,
            "history": self.observations.snapshot() if self.observations else None,
            "session_folder": str(self.folder),
            "video_requested": self.options.record_video,
            "samples_requested": self.options.record_samples,
            "video": video.snapshot() if video else None,
            "samples": samples.snapshot() if samples else None,
            "pipeline": pipe.snapshot() if pipe else None,
            "error": error or self.recording_error,
        }

    def _audit(self, *, final=False, pipe=None, video=None, samples=None, error=None):
        if self.folder is None:
            return
        value = self._summary(pipe, video, samples, final=final, error=error)
        with self._audit_lock:
            write_json(self.folder / "sessao.json", value)
        if final:
            self.last_summary = value

    def _recording_failed(self, error):
        self.recording_error = str(error)
        if self._notified_error != self.recording_error:
            self._notified_error = self.recording_error
            self.notify("recording_error", self.recording_snapshot())

    def _on_result(self, image, data):
        self.observations.submit(data)
        if self.video:
            self.video.submit(image, data)
        error = (self.video.snapshot()["error"] if self.video else None) or (
            self.sampler.error if self.sampler else None
        )
        if error:
            self._recording_failed(error)
        now = time.monotonic()
        if now >= self._next_audit:
            self._next_audit = now + 1
            try:
                self._audit(pipe=self.pipeline, video=self.video, samples=self.sampler)
            except Exception as exc:
                self._recording_failed(f"Falha ao salvar diagnóstico da sessão: {exc}")
        self.notify("frame", (image, data))

    def start(
        self,
        source_factory,
        options: SessionOptions,
        config: AnalyzerConfig | None = None,
    ):
        if not isinstance(options, SessionOptions):
            raise ValueError("SessionOptions esperado")
        with self._lock:
            if self.state != "idle":
                raise RuntimeError("A sessão anterior ainda não foi finalizada")
            self.state = "starting"
            self.options = options
            self.analyzer_config = config or AnalyzerConfig()
            self.provenance = provenance_snapshot(self.analyzer_config)
            self.started_utc = utc_now()
            self.last_summary = None
            self.folder = None
            self._initial_error = self.recording_error = self._notified_error = None
            self._next_audit = 0
            try:
                # Nenhuma captura começa até haver uma pasta gravável e um diagnóstico inicial.
                probe_directory(self.root / "gravacoes")
                self.video = (
                    AnalysisRecorder(
                        self.root / "gravacoes",
                        fps=options.video_fps,
                        log_observations=False,
                    )
                    if options.record_video
                    else None
                )
                self.folder = (
                    self.video.folder
                    if self.video
                    else self.root / "gravacoes" / new_session_id()
                )
                if self.video:
                    self.video.start()
                else:
                    self.folder.mkdir(parents=True, exist_ok=False)
                self.observations = ObservationRecorder(
                    self.folder, video_enabled=options.record_video
                )
                self.observations.start()
                self._audit(video=self.video)
                analyzer = SessionAnalyzer(self.analyzer_config)
                self.sampler = (
                    SampleRecorder(
                        self.root / "capturas",
                        sample_fps=options.sample_fps,
                        session_id=self.folder.name,
                        burst_fps=options.burst_fps,
                    )
                    if options.record_samples
                    else None
                )
                self.pipeline = VisionPipeline(
                    source_factory,
                    analyzer.analyze,
                    self._on_result,
                    on_observation=self.sampler.observe if self.sampler else None,
                    session_id=self.folder.name,
                    roi=options.roi,
                    source_aspect=options.source_aspect,
                    target_fps=options.capture_fps,
                    analysis_fps=options.analysis_fps,
                    max_width=options.max_width,
                    timestamped_analyzer=True,
                    auto_fps=options.auto_fps,
                )
                if self.sampler:
                    self.sampler.start()
                self.pipeline.start()
                self.state = "running"
                self._audit(pipe=self.pipeline, video=self.video, samples=self.sampler)
            except Exception as exc:
                self._initial_error = str(exc)
                self.stop()
                raise

    def stop(self):
        with self._lock:
            if self.state in ("idle", "stopping"):
                return False
            self.state = "stopping"
            pipe, sampler, video = self.pipeline, self.sampler, self.video
            if pipe:
                pipe.stop()
            self._finalizer = threading.Thread(
                target=self._finalize,
                args=(pipe, sampler, video),
                name="BrawllaFinalize",
                daemon=False,
            )
            self._finalizer.start()
            return True

    def _enrich_recording_reports(self, video, sampler):
        if self.folder is None:
            return
        recording = {
            "video_enabled": self.options.record_video,
            "samples_enabled": self.options.record_samples,
            "session_folder": str(self.folder),
            "session_audit": str(self.folder / "sessao.json"),
            "video_file": str(video.video_path) if video and video.video_path else None,
            "video_frames": video.written if video else 0,
            "video_verified": bool(video and video.verified),
            "images_saved": sampler.saved if sampler else 0,
            "samples_folder": str(sampler.folder) if sampler else None,
            "video_error": video.error if video else None,
            "samples_error": sampler.error if sampler else None,
        }
        summary_path = self.folder / "resumo.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["recording"] = recording
            write_json(summary_path, summary)
        elif video is None:
            write_json(
                summary_path,
                {
                    "app_version": VERSION,
                    "read_only": True,
                    "recording": recording,
                    "video_file": None,
                    "video_frames": 0,
                    "events_recorded": 0,
                },
            )
        if self.observations:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update(
                error=self.recording_error
                or self.observations.error
                or (video.error if video else None),
                events_recorded=self.observations.records,
                events_file="analise.jsonl",
                video_mapping_file="video_frames.jsonl" if video else None,
                session_metrics=self.observations.metrics.summary(),
                history=self.observations.snapshot(),
                video_events_processed=video.records if video else 0,
                analyzer_config=asdict(self.analyzer_config),
                provenance=self.provenance,
            )
            write_json(summary_path, summary)
        report_path = self.folder / "relatorio.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["recording"] = recording
            save_report(report, report_path)

    def _finalize(self, pipe, sampler, video):
        errors = []
        if self._initial_error:
            errors.append(self._initial_error)
        if self.recording_error:
            errors.append(self.recording_error)
        for resource in (pipe, sampler, video, self.observations):
            if resource is None:
                continue
            try:
                resource.stop()
            except Exception as exc:
                errors.append(str(exc))
            try:
                resource.join(None)
            except Exception as exc:
                errors.append(str(exc))
        if pipe and pipe.snapshot()["error"]:
            errors.append(pipe.snapshot()["error"])
        if video and video.snapshot()["error"]:
            errors.append(video.snapshot()["error"])
        if self.observations and self.observations.error:
            errors.append(self.observations.error)
        sample_error = sampler.error if sampler else None
        if sample_error:
            errors.append(sample_error)
        self.recording_error = "; ".join(dict.fromkeys(errors)) or None
        try:
            self._enrich_recording_reports(video, sampler)
        except Exception as exc:
            errors.append(f"Falha ao completar o relatório de gravação: {exc}")
        error = "; ".join(dict.fromkeys(errors)) or None
        self.recording_error = error
        try:
            self._audit(
                final=True, pipe=pipe, video=video, samples=sampler, error=error
            )
        except Exception as exc:
            error = "; ".join(
                filter(None, [error, f"Falha ao salvar sessao.json: {exc}"])
            )
            self.recording_error = error
        result = (
            video.snapshot()
            if video
            else {
                "folder": str(self.folder) if self.folder else "",
                "written": 0,
                "records": 0,
                "error": None,
                "video_disabled": True,
            }
        )
        result = {
            **result,
            "error": error,
            "samples": sampler.snapshot() if sampler else None,
            "session_audit": str(self.folder / "sessao.json") if self.folder else None,
        }
        with self._lock:
            self.pipeline = self.sampler = self.video = self.observations = None
            self.state = "idle"
        self.notify("finalized", (result, sample_error))

    def join(self, timeout=None):
        if self._finalizer:
            self._finalizer.join(timeout)
        return self.state == "idle"
