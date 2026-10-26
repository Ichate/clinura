from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import queue
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import anyio
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .assistant import Assistant, AssistantError, local_handover
from .behavior import EyeActivityTracker, MotionTracker
from .capture import BlackFrameError, CameraSource, CaptureError, frame_brightness
from .config import PROJECT_ROOT, CaptureConfig, FaceConfig, HandConfig, ServerConfig, SignalConfig
from .experimental import ExperimentalObservation, build_tongue_roi
from .face import FaceTracker
from .hand import HandTracker
from .pipeline import SignalPipeline
from .recording import SessionRecorder
from .telemetry import EventDetector, TelemetryStore, build_context
from .types import AnalysisResult, FaceObservation

logger = logging.getLogger(__name__)

MODES = ("face", "tongue", "palm")
UI_DIR = PROJECT_ROOT / "app" / "dist"

STARTING = "starting"
RUNNING = "running"
CAMERA_ERROR = "camera_error"
FATAL = "fatal"
STOPPED = "stopped"


@dataclass
class PreviewFrame:
    jpeg: bytes
    frame_id: int
    timestamp: float


@dataclass
class Snapshot:
    observation: FaceObservation | None
    analysis: AnalysisResult
    behavior: dict[str, object]
    mode: str
    experimental: ExperimentalObservation | None
    experimental_analysis: AnalysisResult
    fps: float
    camera_width: int
    camera_height: int
    error: str | None
    state: str
    preview_id: int


def _idle_behavior() -> dict[str, object]:
    return {
        "eye": {"blink_rate": None, "blinks": 0, "openness": None},
        "stability": {"score": 0.0, "motion_px_s": None, "yaw_ratio": 0.0, "roll_deg": 0.0},
    }


class Runtime:
    def __init__(
        self,
        capture: CaptureConfig | None = None,
        face: FaceConfig | None = None,
        signal: SignalConfig | None = None,
        server: ServerConfig | None = None,
        record_dir: str | Path | None = None,
        camera_factory: Callable[[CaptureConfig], CameraSource] | None = None,
        tracker_factory: Callable[[FaceConfig], FaceTracker] | None = None,
        telemetry: TelemetryStore | None = None,
    ) -> None:
        self.capture_config = capture or CaptureConfig()
        self.face_config = face or FaceConfig()
        self.signal_config = signal or SignalConfig()
        self.server_config = server or ServerConfig()
        self.record_dir = Path(record_dir) if record_dir is not None else None
        self._camera_factory = camera_factory or CameraSource
        self._tracker_factory = tracker_factory or FaceTracker
        self._camera: CameraSource | None = None
        self._tracker: FaceTracker | None = None
        self._pipeline = SignalPipeline(self.signal_config)
        self._experimental_config = replace(
            self.signal_config,
            analysis_interval_s=0.2,
            hrv_window_s=30.0,
            min_hrv_window_s=30.0,
            respiration_window_s=45.0,
            min_respiration_window_s=45.0,
            fuse_rois=False,
        )
        self._experimental_pipeline = SignalPipeline(self._experimental_config)
        self._mode = "face"
        self._hand_tracker: HandTracker | None = None
        self._hand_error: str | None = None
        self._experimental: ExperimentalObservation | None = None
        self._eye_tracker = EyeActivityTracker()
        self._motion_tracker = MotionTracker()
        self._behavior: dict[str, object] = _idle_behavior()
        self._recorder: SessionRecorder | None = None
        self._commands: queue.SimpleQueue[tuple[str, str | None]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._preview_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._preview: PreviewFrame | None = None
        self._observation: FaceObservation | None = None
        self._error: str | None = None
        self._state = STOPPED
        self._started = False
        self._started_at = 0.0
        self._preview_id = 0
        self._fps = 0.0
        self._camera_size = (0, 0)
        self._camera_index = self.capture_config.camera_index
        self._payload_cache: tuple[float, dict[str, object]] | None = None
        self._payload_lock = threading.Lock()
        self.telemetry = telemetry or TelemetryStore(self.server_config.telemetry_path)
        self._detector = EventDetector(self.telemetry)
        self._last_telemetry = 0.0

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def state(self) -> str:
        return self._state

    @property
    def running(self) -> bool:
        return self._state == RUNNING

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._started_at = time.monotonic()
        self._stop.clear()
        self._state = STARTING
        self._thread = threading.Thread(target=self._supervise, name="clinura-runtime", daemon=True)
        self._thread.start()
        self._preview_thread = threading.Thread(
            target=self._preview_loop, name="clinura-preview", daemon=True
        )
        self._preview_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=2.0)
            self._preview_thread = None
        self._close_camera()
        if self._hand_tracker is not None:
            self._hand_tracker.close()
            self._hand_tracker = None
        self._detector.close_all()
        self.telemetry.flush()
        self._started = False
        self._state = STOPPED

    def _set_error(self, state: str, message: str | None) -> None:
        self._state = state
        self._error = message

    def _close_camera(self) -> None:
        camera, self._camera = self._camera, None
        if camera is not None:
            camera.stop()

    def _supervise(self) -> None:
        try:
            self._tracker = self._tracker_factory(self.face_config)
        except Exception as exc:
            logger.exception("Face landmarker initialization failed")
            self._set_error(
                FATAL,
                f"Face tracking model could not be loaded ({exc}). Run `make model` and restart.",
            )
            return

        try:
            candidates = self._camera_candidates()
            position = 0
            failures: dict[int, str] = {}
            while not self._stop.is_set():
                index = candidates[position % len(candidates)]
                config = replace(self.capture_config, camera_index=index)
                try:
                    camera = self._camera_factory(config)
                    camera.start()
                except (CaptureError, RuntimeError, OSError) as exc:
                    logger.warning("Camera %d unavailable: %s", index, exc)
                    failures[index] = str(exc)
                    position += 1
                    if position % len(candidates) == 0:
                        self._set_error(CAMERA_ERROR, self._camera_failure_message(failures))
                        self._stop.wait(self.server_config.camera_retry_s)
                    continue

                if index != self.capture_config.camera_index:
                    logger.info(
                        "Camera %d is unusable; using camera %d instead. Set CLINURA_CAMERA=%d "
                        "to start with it directly.",
                        self.capture_config.camera_index,
                        index,
                        index,
                    )
                failures.clear()
                self._camera = camera
                with self._lock:
                    self._camera_size = camera.size
                    self._camera_index = index
                self._set_error(RUNNING, None)
                self._open_recorder(camera)
                try:
                    self._process(camera)
                except BlackFrameError as exc:
                    logger.warning("%s Trying the next camera.", exc)
                    failures[index] = str(exc)
                    position += 1
                    self._set_error(CAMERA_ERROR, f"{exc} Looking for another camera…")
                except CaptureError as exc:
                    logger.warning("Camera lost: %s", exc)
                    self._set_error(CAMERA_ERROR, f"{exc} Reconnecting…")
                except Exception as exc:
                    logger.exception("Processing loop failed; restarting")
                    self._set_error(CAMERA_ERROR, f"Processing failed ({exc}). Restarting…")
                finally:
                    self._close_camera()
                    self._reset_signal_state()
                if not self._stop.is_set():
                    self._stop.wait(self.server_config.camera_retry_s)
        finally:
            if self._recorder is not None:
                self._recorder.close()
                self._recorder = None
            if self._tracker is not None:
                self._tracker.close()
                self._tracker = None

    def _camera_candidates(self) -> list[int]:
        first = self.capture_config.camera_index
        if not self.capture_config.auto_select:
            return [first]
        others = [i for i in range(self.capture_config.max_camera_index + 1) if i != first]
        return [first, *others]

    def _camera_failure_message(self, failures: dict[int, str]) -> str:
        if failures and all("black frames" in message for message in failures.values()):
            return (
                "Every camera is sending black frames. Check that the lens isn't covered and "
                "that your terminal has camera access in System Settings > Privacy & Security "
                "> Camera, then restart Clinura."
            )
        preferred = failures.get(self.capture_config.camera_index)
        return preferred or next(iter(failures.values()), "No camera is available.")

    def _open_recorder(self, camera: CameraSource) -> None:
        if self.record_dir is None or self._recorder is not None:
            return
        try:
            self._recorder = SessionRecorder(
                self.record_dir,
                fps=self.capture_config.fps,
                frame_size=camera.size,
                capture=self.capture_config,
                face=self.face_config,
                signal=self.signal_config,
                name=datetime.now().strftime("session-%Y%m%d-%H%M%S"),
            )
            logger.info("Recording session to %s", self._recorder.output_dir)
        except (RuntimeError, OSError):
            logger.exception("Session recorder could not start; continuing without recording")
            self.record_dir = None

    def set_mode(self, mode: str) -> None:
        if mode in MODES:
            self._commands.put(("mode", mode))

    def reset(self) -> None:
        self._commands.put(("reset", None))

    def _apply_commands(self) -> None:
        while True:
            try:
                command, value = self._commands.get_nowait()
            except queue.Empty:
                return
            if command == "reset":
                self._reset_signal_state()
            elif command == "mode" and value is not None and value != self._mode:
                self._switch_mode(value)

    def _switch_mode(self, mode: str) -> None:
        if mode == "palm" and self._hand_tracker is None:
            try:
                self._hand_tracker = HandTracker(HandConfig())
                self._hand_error = None
            except Exception as exc:
                logger.exception("Hand landmarker initialization failed")
                self._hand_error = f"Palm tracking unavailable: {exc}"
        self._mode = mode
        self._experimental_pipeline.reset()
        with self._lock:
            self._experimental = None

    def _reset_signal_state(self) -> None:
        self._pipeline.reset()
        self._experimental_pipeline.reset()
        self._eye_tracker.reset()
        self._motion_tracker.reset()
        with self._lock:
            self._observation = None
            self._experimental = None
            self._behavior = _idle_behavior()

    def _process(self, camera: CameraSource) -> None:
        assert self._tracker is not None
        tracker_failures = 0
        last_frame_at = time.monotonic()
        last_bright_at = time.monotonic()
        dark_limit = self.capture_config.black_frame_timeout_s * 2.5
        while not self._stop.is_set():
            self._apply_commands()
            try:
                timestamp, frame = camera.read(timeout=0.5)
            except queue.Empty:
                stalled = time.monotonic() - last_frame_at > self.capture_config.stall_timeout_s
                if camera.failed or stalled:
                    raise CaptureError("The camera stopped delivering frames.") from None
                continue
            last_frame_at = time.monotonic()
            if frame_brightness(frame) >= self.capture_config.black_level:
                last_bright_at = last_frame_at
            elif last_frame_at - last_bright_at > dark_limit:
                raise BlackFrameError(
                    f"Camera {camera.config.camera_index} went dark for {dark_limit:.0f} s."
                )

            try:
                observation = self._tracker.detect(frame, timestamp)
                tracker_failures = 0
            except Exception:
                tracker_failures += 1
                if tracker_failures == 1 or tracker_failures % 100 == 0:
                    logger.exception("Face tracking failed (%d consecutive)", tracker_failures)
                if tracker_failures >= 300:
                    raise
                continue

            self._process_frame(frame, timestamp, observation)
            with self._lock:
                self._fps = camera.measured_fps

    def _process_frame(
        self, frame: np.ndarray, timestamp: float, observation: FaceObservation
    ) -> None:
        self._pipeline.set_face_status(
            observation.detected,
            observation.quality,
            observation.reason,
            timestamp,
            pose_ok=observation.pose_ok,
        )
        rgb = observation.combined_rgb
        if (
            observation.detected
            and observation.pose_ok
            and rgb is not None
            and observation.quality >= self.signal_config.min_sample_quality
        ):
            usable = {name: roi for name, roi in observation.rois.items() if roi.pixel_count > 0}
            self._pipeline.add_sample(
                timestamp,
                rgb,
                observation.quality,
                observation.face_width_px,
                observation.roi_pixels,
                roi_rgb={name: roi.mean_rgb for name, roi in usable.items()},
                roi_pixels_by_name={name: roi.pixel_count for name, roi in usable.items()},
            )

        if self._recorder is not None:
            try:
                self._recorder.write(frame, observation, timestamp)
            except Exception:
                logger.exception("Recording frame failed; recording stopped")
                self._recorder.close()
                self._recorder = None

        self._pipeline.analyze(timestamp)
        experimental = self._process_experimental(frame, timestamp, observation)

        if observation.detected:
            eye = self._eye_tracker.update(
                observation.blink_left, observation.blink_right, timestamp
            )
        else:
            eye = _idle_behavior()["eye"]
        stability = self._motion_tracker.update(
            timestamp,
            observation.yaw_ratio if observation.detected else 0.0,
            observation.roll_deg if observation.detected else 0.0,
            observation.bbox if observation.detected else None,
        )
        with self._lock:
            self._observation = observation
            self._behavior = {"eye": eye, "stability": stability}
            self._experimental = experimental
        self._record_telemetry(observation, eye, stability)

    def _record_telemetry(
        self, observation: FaceObservation, eye: dict[str, object], stability: dict[str, object]
    ) -> None:
        now = time.time()
        if now - self._last_telemetry < 1.0:
            return
        self._last_telemetry = now
        result = self._pipeline.latest
        openness = eye.get("openness")
        sample = {
            "ts": now,
            "bpm": _number(result.bpm, 1),
            "provisional_bpm": _number(result.provisional_bpm, 1),
            "rmssd_ms": _number(result.rmssd_ms, 1) if result.hrv_ready else None,
            "sdnn_ms": _number(result.hrv_sdnn_ms, 1) if result.hrv_ready else None,
            "respiration_bpm": _number(result.respiration_bpm, 1)
            if result.respiration_ready
            else None,
            "snr_db": _number(result.snr_db, 2),
            "confidence": _number(result.confidence, 3),
            "status": result.status,
            "stability": _number(stability.get("score"), 1) if observation.detected else None,
            "motion_px_s": _number(stability.get("motion_px_s"), 1),
            "face_detected": int(observation.detected),
            "eyes_closed": None if openness is None else round(1.0 - float(openness), 3),
        }
        try:
            self.telemetry.record(sample)
            self._detector.update(sample)
        except Exception:
            logger.exception("Telemetry write failed")

    def _process_experimental(
        self, frame: np.ndarray, timestamp: float, observation: FaceObservation
    ) -> ExperimentalObservation | None:
        mode = self._mode
        if mode == "face":
            return None
        if mode == "tongue":
            if observation.detected and observation.pose_ok and observation.landmarks is not None:
                experimental = build_tongue_roi(frame, observation.landmarks, observation)
            else:
                experimental = ExperimentalObservation(
                    mode="tongue", active=False, reason="Face the camera and open your mouth"
                )
        elif self._hand_tracker is not None:
            experimental = self._hand_tracker.detect(frame, timestamp)
        else:
            experimental = ExperimentalObservation(
                mode="palm", active=False, reason=self._hand_error or "Palm tracking unavailable"
            )

        if experimental.active and experimental.roi is not None:
            self._experimental_pipeline.set_face_status(True, experimental.quality, None, timestamp)
            self._experimental_pipeline.add_sample(
                timestamp,
                experimental.roi.mean_rgb,
                experimental.quality,
                0.0,
                experimental.roi.pixel_count,
            )
        else:
            self._experimental_pipeline.set_face_status(False, 0.0, experimental.reason, timestamp)
        self._experimental_pipeline.analyze(timestamp)
        return experimental

    def _encode_preview(self, frame: np.ndarray) -> bytes | None:
        width = int(self.server_config.preview_width)
        if frame.shape[1] > width:
            scale = width / float(frame.shape[1])
            frame = cv2.resize(
                frame,
                (width, max(2, int(round(frame.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.server_config.preview_jpeg_quality]
        )
        return encoded.tobytes() if ok else None

    def _preview_loop(self) -> None:
        interval = 1.0 / max(self.server_config.preview_fps, 1.0)
        next_tick = time.perf_counter()
        last_timestamp = -1.0
        while not self._stop.is_set():
            camera = self._camera
            latest = camera.latest() if camera is not None else None
            if latest is not None and latest[0] != last_timestamp:
                timestamp, frame = latest
                last_timestamp = timestamp
                try:
                    jpeg = self._encode_preview(frame)
                except cv2.error:
                    logger.debug("Preview encoding failed", exc_info=True)
                    jpeg = None
                if jpeg is not None:
                    with self._lock:
                        self._preview_id += 1
                        self._preview = PreviewFrame(
                            jpeg=jpeg, frame_id=self._preview_id, timestamp=timestamp
                        )
            elif camera is None and self._preview is not None:
                with self._lock:
                    self._preview = None
            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_tick = time.perf_counter()

    def preview_frame(self) -> PreviewFrame | None:
        with self._lock:
            return self._preview

    def snapshot(self) -> Snapshot:
        with self._lock:
            observation = self._observation
            preview = self._preview
            preview_id = preview.frame_id if preview is not None else 0
            fps = self._fps if self._camera is not None else 0.0
            behavior = dict(self._behavior)
            experimental = self._experimental
            camera_size = self._camera_size
        return Snapshot(
            observation=observation,
            analysis=self._pipeline.latest,
            behavior=behavior,
            mode=self._mode,
            experimental=experimental,
            experimental_analysis=self._experimental_pipeline.latest,
            fps=fps,
            camera_width=camera_size[0],
            camera_height=camera_size[1],
            error=self._error,
            state=self._state,
            preview_id=preview_id,
        )

    def hello_payload(self) -> dict[str, object]:
        with self._lock:
            width, height = self._camera_size
            fps = self._fps
        return _clean(
            {
                "type": "hello",
                "version": __version__,
                "state": self._state,
                "camera": {"width": width, "height": height, "fps": round(fps, 2)},
                "model": "mediapipe-face-landmarker",
                "modes": list(MODES),
            }
        )

    def health_payload(self) -> dict[str, object]:
        with self._lock:
            width, height = self._camera_size
            fps = self._fps
        return _clean(
            {
                "ok": self._state == RUNNING,
                "state": self._state,
                "running": self.running,
                "error": self._error,
                "version": __version__,
                "mode": self._mode,
                "uptime_s": round(time.monotonic() - self._started_at, 1) if self._started else 0,
                "camera": {
                    "width": width,
                    "height": height,
                    "fps": round(fps, 2),
                    "index": self._camera_index,
                },
                "recording": self._recorder is not None,
            }
        )

    def metrics_payload(self) -> dict[str, object]:
        max_age = 0.5 / max(self.server_config.metrics_hz, 1.0)
        now = time.monotonic()
        with self._payload_lock:
            if self._payload_cache is not None and now - self._payload_cache[0] < max_age:
                return self._payload_cache[1]
            payload = _clean(self._build_metrics())
            self._payload_cache = (now, payload)
            return payload

    def _build_metrics(self) -> dict[str, object]:
        snapshot = self.snapshot()
        result = snapshot.analysis
        observation = snapshot.observation
        error = snapshot.error
        engine_ok = snapshot.state == RUNNING
        if error:
            status, message = "error", error
        elif not engine_ok:
            status, message = "acquiring", "Starting the camera…"
        else:
            status, message = result.status, result.message

        live = engine_ok and observation is not None
        face_detected = bool(live and observation.detected)
        rois: list[dict[str, object]] = []
        face_bbox = None
        face_contour = None
        if face_detected and observation is not None:
            if (
                observation.bbox is not None
                and snapshot.camera_width > 0
                and snapshot.camera_height > 0
            ):
                bx, by, bw, bh = observation.bbox
                face_bbox = [
                    round(bx / snapshot.camera_width, 4),
                    round(by / snapshot.camera_height, 4),
                    round(bw / snapshot.camera_width, 4),
                    round(bh / snapshot.camera_height, 4),
                ]
            if observation.face_contour is not None and len(observation.face_contour) > 0:
                face_contour = _contour(observation.face_contour)
            rois = [
                {"name": name, "pixels": int(roi.pixel_count), "contour": _contour(roi.contour)}
                for name, roi in observation.rois.items()
            ]

        if not engine_ok:
            result = AnalysisResult(timestamp=result.timestamp, status=status, message=message)

        base = float(result.timestamp)
        beats: list[float] = []
        if result.beats is not None and len(result.beats) > 0:
            beats = [round(float(b - base), 3) for b in result.beats]
            beats = [b for b in beats if -result.window_s - 2.0 <= b <= 2.0]
        ibi_times: list[float] = []
        if result.ibi_times is not None and len(result.ibi_times) > 0:
            ibi_times = [round(float(t - base), 3) for t in result.ibi_times]
            ibi_times = [t for t in ibi_times if -62.0 <= t <= 2.0]
        ibis_ms: list[float] = []
        if result.ibis_ms is not None and len(result.ibis_ms) > 0:
            ibis_ms = [round(float(v), 1) for v in result.ibis_ms]

        waveform_t, waveform_y = _downsample_pair(result.waveform_t, result.waveform_y, 360)
        respiration_t, respiration_y = _downsample_pair(
            result.respiration_t, result.respiration_y, 180
        )
        roi_quality = [
            {
                "name": str(item.get("name", "")),
                "bpm": _number(item.get("bpm"), 1),
                "snr_db": _number(item.get("snr_db"), 2),
                "pixels": int(item.get("pixels", 0)),
            }
            for item in result.roi_quality
        ]

        return {
            "type": "metrics",
            "timestamp": round(base, 4),
            "engine_state": snapshot.state,
            "mode": snapshot.mode,
            "status": status,
            "message": message,
            "error": error,
            "bpm": _number(result.bpm, 1),
            "provisional_bpm": _number(result.provisional_bpm, 1),
            "snr_db": _number(result.snr_db, 2),
            "rmssd_ms": _number(result.rmssd_ms, 1),
            "hrv_ready": bool(result.hrv_ready),
            "hrv_progress": _number(result.hrv_progress, 3) or 0.0,
            "confidence": _number(result.confidence, 3) or 0.0,
            "fs": _number(result.fs, 2),
            "window_s": _number(result.window_s, 2) or 0.0,
            "sample_count": int(result.sample_count),
            "ibi_count": int(result.ibi_count),
            "analysis_ms": _number(result.analysis_ms, 2),
            "face_detected": face_detected,
            "pose_ok": bool(face_detected and observation.pose_ok),
            "face_reason": observation.reason if live else None,
            "face_bbox": face_bbox,
            "face_contour": face_contour,
            "face_width_px": round(float(observation.face_width_px), 1) if live else 0.0,
            "roi_pixels": int(observation.roi_pixels) if live else 0,
            "face_quality": round(float(observation.quality), 3) if live else 0.0,
            "yaw_ratio": round(float(observation.yaw_ratio), 3) if face_detected else None,
            "roll_deg": round(float(observation.roll_deg), 1) if face_detected else None,
            "rois": rois,
            "eye": snapshot.behavior.get("eye", {}),
            "stability": snapshot.behavior.get("stability", {}),
            "camera": {
                "width": snapshot.camera_width if engine_ok else 0,
                "height": snapshot.camera_height if engine_ok else 0,
                "fps": round(snapshot.fps, 2),
                "index": self._camera_index,
            },
            "waveform": {"t": waveform_t, "y": waveform_y},
            "raw": _downsample_rgb(result.raw_t, result.raw_rgb, 360),
            "beats": beats,
            "psd": _downsample_psd(result.psd_f, result.psd_p),
            "roi_quality": roi_quality,
            "lighting": {
                "mean": _number(result.luma_mean, 1),
                "drift": _number(result.luma_std, 2),
            },
            "hrv": {
                "sdnn_ms": _number(result.hrv_sdnn_ms, 1),
                "pnn50": _number(result.hrv_pnn50, 1),
                "mean_ibi_ms": _number(result.hrv_mean_ibi_ms, 1),
                "ibi_times": ibi_times,
                "ibis_ms": ibis_ms,
            },
            "respiration": {
                "bpm": _number(result.respiration_bpm, 1),
                "snr_db": _number(result.respiration_snr_db, 2),
                "ready": bool(result.respiration_ready),
                "progress": _number(result.respiration_progress, 3) or 0.0,
                "t": respiration_t,
                "y": respiration_y,
            },
            "experimental": self._experimental_payload(snapshot) if engine_ok else None,
        }

    def _experimental_payload(self, snapshot: Snapshot) -> dict[str, object] | None:
        if snapshot.mode == "face":
            return None
        experimental = snapshot.experimental
        result = snapshot.experimental_analysis
        roi = experimental.roi if experimental is not None and experimental.active else None
        wave_t, wave_y = _downsample_pair(result.waveform_t, result.waveform_y, 240)
        return {
            "mode": snapshot.mode,
            "active": roi is not None,
            "status": result.status if roi is not None else "no_roi",
            "message": (
                experimental.reason
                if experimental is not None and experimental.reason
                else result.message
            ),
            "bpm": _number(result.bpm, 1),
            "provisional_bpm": _number(result.provisional_bpm, 1),
            "snr_db": _number(result.snr_db, 2),
            "confidence": _number(result.confidence, 3) or 0.0,
            "quality": None if experimental is None else round(float(experimental.quality), 3),
            "handedness": None if experimental is None else experimental.handedness,
            "pixels": 0 if roi is None else int(roi.pixel_count),
            "roi": None
            if roi is None
            else {
                "name": roi.name,
                "pixels": int(roi.pixel_count),
                "contour": _contour(roi.contour),
            },
            "waveform": {"t": wave_t, "y": wave_y},
            "psd": _downsample_psd(result.psd_f, result.psd_p),
            "window_s": _number(result.window_s, 2),
            "analysis_ms": _number(result.analysis_ms, 2),
        }


def _number(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    number = float(value)
    return round(number, digits) if math.isfinite(number) else None


def _clean(value: object) -> object:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _contour(points: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 4), round(float(y), 4)] for x, y in points]


def _downsample_pair(
    x: np.ndarray | None, y: np.ndarray | None, max_points: int
) -> tuple[list[float], list[float]]:
    if x is None or y is None or len(x) == 0 or len(y) == 0:
        return [], []
    count = min(len(x), len(y))
    x = np.asarray(x[:count], dtype=np.float64)
    y = np.asarray(y[:count], dtype=np.float64)
    step = max(1, int(np.ceil(count / max_points)))
    return (
        [round(float(v), 3) for v in x[::step]],
        [round(float(v), 4) for v in y[::step]],
    )


def _downsample_psd(
    freqs: np.ndarray | None,
    power: np.ndarray | None,
    low_hz: float = 0.5,
    high_hz: float = 3.5,
    max_points: int = 96,
) -> dict[str, list[float]]:
    if freqs is None or power is None or len(freqs) == 0 or len(power) == 0:
        return {"f": [], "p": []}
    count = min(len(freqs), len(power))
    freqs = np.asarray(freqs[:count], dtype=np.float64)
    power = np.asarray(power[:count], dtype=np.float64)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    freqs = freqs[mask]
    power = power[mask]
    if freqs.size == 0:
        return {"f": [], "p": []}
    step = max(1, int(np.ceil(freqs.size / max_points)))
    return {
        "f": [round(float(v), 3) for v in freqs[::step]],
        "p": [float(f"{float(v):.4g}") for v in power[::step]],
    }


def _downsample_rgb(
    x: np.ndarray | None, rgb: np.ndarray | None, max_points: int
) -> dict[str, list[float]]:
    if x is None or rgb is None or len(x) == 0 or len(rgb) == 0:
        return {"t": [], "r": [], "g": [], "b": []}
    count = min(len(x), len(rgb))
    x = np.asarray(x[:count], dtype=np.float64)
    rgb = np.asarray(rgb[:count], dtype=np.float64)
    step = max(1, int(np.ceil(count / max_points)))
    return {
        "t": [round(float(v), 3) for v in x[::step]],
        "r": [round(float(v), 2) for v in rgb[::step, 0]],
        "g": [round(float(v), 2) for v in rgb[::step, 1]],
        "b": [round(float(v), 2) for v in rgb[::step, 2]],
    }


def _parse_command(text: str) -> dict[str, object] | None:
    try:
        message = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return message if isinstance(message, dict) else None


class HandoverRequest(BaseModel):
    hours: float = Field(default=6.0, ge=0.05, le=72.0)
    bed: str | None = Field(default=None, max_length=40)
    patient: str | None = Field(default=None, max_length=80)


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    hours: float = Field(default=12.0, ge=0.05, le=72.0)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class NoteRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    detail: str | None = Field(default=None, max_length=500)


def create_app(
    runtime: Runtime, ui_dir: Path | None = UI_DIR, assistant: Assistant | None = None
) -> FastAPI:
    copilot = assistant or Assistant()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            await anyio.to_thread.run_sync(runtime.stop)

    app = FastAPI(title="Clinura", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://{host}:{port}"
            for host in ("localhost", "127.0.0.1")
            for port in (3000, 5173, 4173)
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return runtime.health_payload()

    @app.post("/api/reset")
    def reset() -> dict[str, bool]:
        runtime.reset()
        return {"ok": True}

    @app.get("/api/assistant")
    def assistant_status() -> dict[str, object]:
        return copilot.status()

    @app.get("/api/telemetry")
    def telemetry(hours: float = Query(default=1.0, ge=0.05, le=72.0)) -> dict[str, object]:
        return _clean(build_context(runtime.telemetry, hours))

    @app.get("/api/events")
    def events(hours: float = Query(default=12.0, ge=0.05, le=168.0)) -> list[dict[str, object]]:
        since = time.time() - hours * 3600.0
        return _clean([event.to_dict() for event in runtime.telemetry.events(since)])

    @app.post("/api/events")
    def add_note(note: NoteRequest) -> dict[str, object]:
        event_id = runtime.telemetry.add_event(
            "note",
            "info",
            note.title.strip(),
            detail=(note.detail or "").strip() or None,
            end_ts=time.time(),
            source="manual",
        )
        return {"ok": True, "id": event_id}

    @app.post("/api/handover")
    async def handover(request: HandoverRequest) -> dict[str, object]:
        context = await anyio.to_thread.run_sync(build_context, runtime.telemetry, request.hours)
        generated_by = copilot.config.model
        if copilot.config.configured:
            try:
                text = await copilot.handover(context, request.bed, request.patient)
            except AssistantError as exc:
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        else:
            text = local_handover(context, request.bed, request.patient)
            generated_by = "local template"
        return _clean(
            {
                "text": text,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "generated_by": generated_by,
                "window": context["window"],
                "coverage": context["coverage"],
            }
        )

    @app.post("/api/ask")
    async def ask(request: AskRequest) -> dict[str, object]:
        context = await anyio.to_thread.run_sync(build_context, runtime.telemetry, request.hours)
        try:
            answer = await copilot.ask(
                request.question.strip(),
                [turn.model_dump() for turn in request.history],
                context,
            )
        except AssistantError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return {"answer": answer, "model": copilot.config.model}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()

        async def send_metrics() -> None:
            interval = 1.0 / max(runtime.server_config.metrics_hz, 0.5)
            while True:
                payload = await anyio.to_thread.run_sync(runtime.metrics_payload)
                await websocket.send_text(json.dumps(payload, allow_nan=False))
                await asyncio.sleep(interval)

        async def send_frames() -> None:
            interval = 1.0 / max(runtime.server_config.preview_fps, 1.0)
            last_id = -1
            while True:
                preview = runtime.preview_frame()
                if preview is not None and preview.frame_id != last_id:
                    await websocket.send_bytes(preview.jpeg)
                    last_id = preview.frame_id
                await asyncio.sleep(interval)

        async def receive_commands() -> None:
            while True:
                message = _parse_command(await websocket.receive_text())
                if message is None:
                    continue
                kind = message.get("type")
                if kind == "set_mode":
                    runtime.set_mode(str(message.get("mode", "face")))
                elif kind == "reset":
                    runtime.reset()

        try:
            await websocket.send_text(json.dumps(runtime.hello_payload(), allow_nan=False))
        except (WebSocketDisconnect, RuntimeError):
            return

        async with anyio.create_task_group() as group:

            async def run(loop: Callable[[], Awaitable[None]]) -> None:
                try:
                    await loop()
                except (WebSocketDisconnect, RuntimeError):
                    pass
                except Exception:
                    logger.exception("WebSocket session failed")
                finally:
                    group.cancel_scope.cancel()

            group.start_soon(run, send_metrics)
            group.start_soon(run, send_frames)
            group.start_soon(run, receive_commands)

        with contextlib.suppress(Exception):
            await websocket.close()

    if ui_dir is not None and (ui_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
    else:

        @app.get("/")
        def index() -> dict[str, str]:
            return {
                "message": "Clinura engine is running, but the UI is not built. "
                "Run `python main.py` or `make build-ui`.",
                "health": "/api/health",
            }

    return app
