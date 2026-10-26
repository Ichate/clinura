from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import uvicorn

from .capture import CameraSource, CaptureError
from .config import CaptureConfig, FaceConfig, ServerConfig, SignalConfig
from .face import FaceTracker
from .pipeline import SignalPipeline
from .recording import SessionRecorder
from .replay import compare_to_reference, replay_session
from .server import Runtime, create_app

logger = logging.getLogger(__name__)

ROI_COLORS = {
    "forehead": (216, 231, 124),
    "right_cheek": (255, 180, 138),
    "left_cheek": (107, 184, 255),
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _capture_config(args: argparse.Namespace) -> CaptureConfig:
    return CaptureConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mirror=not args.no_mirror,
    )


def _draw_hud(frame: np.ndarray, result: object, observation: object, fps: float) -> None:
    height, width = frame.shape[:2]
    panel_h = 150
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - panel_h), (width, height), (8, 12, 15), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    cv2.line(frame, (0, height - panel_h), (width, height - panel_h), (52, 70, 76), 1)

    bpm = getattr(result, "bpm", None)
    bpm_text = "--" if bpm is None else f"{bpm:.0f}"
    cv2.putText(
        frame,
        bpm_text,
        (34, height - 52),
        cv2.FONT_HERSHEY_DUPLEX,
        2.6,
        (240, 246, 244),
        3,
        cv2.LINE_AA,
    )
    bpm_width = cv2.getTextSize(bpm_text, cv2.FONT_HERSHEY_DUPLEX, 2.6, 3)[0][0]
    cv2.putText(
        frame,
        "BPM",
        (48 + bpm_width, height - 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (122, 158, 150),
        2,
        cv2.LINE_AA,
    )

    status = getattr(result, "status", "unknown")
    message = getattr(result, "message", "")
    color = (120, 230, 190) if status == "good" else (80, 180, 255)
    if status in ("low_signal", "no_face", "error"):
        color = (120, 140, 255)
    cv2.circle(frame, (36, height - 96), 5, color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        status.replace("_", " ").upper(),
        (50, height - 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message[:80],
        (50, height - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 196, 192),
        1,
        cv2.LINE_AA,
    )

    rmssd = getattr(result, "rmssd_ms", None)
    hrv_text = "HRV --" if rmssd is None else f"HRV {rmssd:.0f} ms"
    snr = getattr(result, "snr_db", None)
    snr_text = "SNR --" if snr is None else f"SNR {snr:.1f} dB"
    cv2.putText(
        frame,
        hrv_text,
        (width - 480, height - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (206, 214, 210),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        snr_text,
        (width - 480, height - 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (206, 214, 210),
        2,
        cv2.LINE_AA,
    )
    face_width = getattr(observation, "face_width_px", 0.0)
    roi_pixels = getattr(observation, "roi_pixels", 0)
    cv2.putText(
        frame,
        f"face {face_width:.0f}px · roi {roi_pixels}px · {fps:.1f} fps",
        (width - 480, height - 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (122, 158, 150),
        1,
        cv2.LINE_AA,
    )

    waveform = getattr(result, "waveform_y", None)
    if waveform is not None and len(waveform) > 8:
        area_x0 = width - 610
        area_x1 = width - 520
        area_y0 = height - 134
        area_y1 = height - 24
        x_axis = np.linspace(area_x0, area_x1, len(waveform))
        values = np.asarray(waveform, dtype=np.float64)
        spread = float(np.max(np.abs(values))) or 1.0
        y_axis = (area_y0 + area_y1) / 2.0 - values / spread * (area_y1 - area_y0) * 0.42
        points = np.column_stack([x_axis, y_axis]).astype(np.int32)
        cv2.polylines(frame, [points], False, (130, 235, 210), 2, cv2.LINE_AA)

    cv2.putText(
        frame,
        "Q to quit",
        (width - 120, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (140, 156, 152),
        1,
        cv2.LINE_AA,
    )


def _draw_rois(frame: np.ndarray, observation: object) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    face_contour = getattr(observation, "face_contour", None)
    if face_contour is not None and len(face_contour) > 0:
        points = np.column_stack([face_contour[:, 0] * width, face_contour[:, 1] * height]).astype(
            np.int32
        )
        cv2.polylines(frame, [points], True, (210, 220, 216), 1, cv2.LINE_AA)
    rois = getattr(observation, "rois", {})
    for name, roi in rois.items():
        if len(roi.contour) < 3:
            continue
        points = np.column_stack([roi.contour[:, 0] * width, roi.contour[:, 1] * height]).astype(
            np.int32
        )
        color = ROI_COLORS.get(name, (180, 220, 200))
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
        center = points.mean(axis=0).astype(int)
        label = name.replace("_", " ")
        cv2.putText(
            frame,
            label,
            (int(center[0]) - 30, int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 246, 244),
            1,
            cv2.LINE_AA,
        )
    frame[:] = cv2.addWeighted(overlay, 0.16, frame, 0.84, 0)


def cmd_serve(args: argparse.Namespace) -> int:
    capture = _capture_config(args)
    runtime = Runtime(
        capture=capture,
        server=ServerConfig(host=args.host, port=args.port),
        record_dir=args.record_dir,
    )
    app = create_app(runtime)
    print(f"Clinura engine: http://{args.host}:{args.port}")
    print("UI dev server:  cd app && npm run dev")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    capture = CameraSource(_capture_config(args))
    tracker = FaceTracker(FaceConfig())
    pipeline = SignalPipeline(SignalConfig())
    try:
        capture.start()
        print("Press Q in the preview window to quit.")
        while True:
            timestamp, frame = capture.read(timeout=2.0)
            observation = tracker.detect(frame, timestamp)
            pipeline.set_face_status(
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
                and observation.quality >= 0.08
            ):
                roi_rgb = {
                    name: roi.mean_rgb
                    for name, roi in observation.rois.items()
                    if roi.pixel_count > 0
                }
                roi_pixels_by_name = {
                    name: roi.pixel_count
                    for name, roi in observation.rois.items()
                    if roi.pixel_count > 0
                }
                pipeline.add_sample(
                    timestamp,
                    rgb,
                    observation.quality,
                    observation.face_width_px,
                    observation.roi_pixels,
                    roi_rgb=roi_rgb,
                    roi_pixels_by_name=roi_pixels_by_name,
                )
            result = pipeline.analyze(timestamp)
            if observation.detected:
                _draw_rois(frame, observation)
            _draw_hud(frame, result, observation, capture.measured_fps)
            cv2.imshow("Clinura | live signal", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        capture.stop()
        tracker.close()
        cv2.destroyAllWindows()
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    capture = CameraSource(_capture_config(args))
    tracker = FaceTracker(FaceConfig())
    recorder: SessionRecorder | None = None
    try:
        capture.start()
        recorder = SessionRecorder(
            args.output,
            fps=args.fps,
            frame_size=capture.size,
            capture=_capture_config(args),
            face=FaceConfig(),
            signal=SignalConfig(),
            name=datetime.now().strftime("session-%Y%m%d-%H%M%S"),
        )
        print(f"Recording {args.seconds}s to {recorder.output_dir}")
        started = time.perf_counter()
        last_report = started
        while time.perf_counter() - started < args.seconds:
            timestamp, frame = capture.read(timeout=2.0)
            observation = tracker.detect(frame, timestamp)
            recorder.write(frame, observation, timestamp)
            now = time.perf_counter()
            if now - last_report >= 2.0:
                elapsed = now - started
                print(
                    f"  {elapsed:5.1f}s  {recorder.frame_count} frames  "
                    f"{capture.measured_fps:.1f} fps  "
                    f"face={'yes' if observation.detected else 'no'}"
                )
                last_report = now
    except KeyboardInterrupt:
        print("\nStopping recording.")
    finally:
        capture.stop()
        tracker.close()
        if recorder is not None:
            path = recorder.close()
            print(f"Saved session: {path}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    session = Path(args.session)
    results = replay_session(session, recompute=args.recompute)
    if args.reference:
        metrics = compare_to_reference(
            session / "analysis.jsonl", args.reference, tolerance_bpm=args.tolerance
        )
        print(
            "reference: "
            f"n={metrics['n']:.0f} mae={metrics['mae']:.2f} bpm "
            f"rmse={metrics['rmse']:.2f} bpm "
            f"within ±{args.tolerance:g} bpm={metrics['within_tolerance_pct']:.1f}%"
        )
    return 0 if results else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clinura", description="Clinura rPPG engine")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_camera_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--camera", type=int, default=0, help="camera index")
        command.add_argument("--width", type=int, default=1920)
        command.add_argument("--height", type=int, default=1080)
        command.add_argument("--fps", type=int, default=30)
        command.add_argument("--no-mirror", action="store_true", help="disable selfie mirroring")

    serve = sub.add_parser("serve", help="run the WebSocket engine and UI")
    add_camera_options(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=3000)
    serve.add_argument("--record-dir", type=Path, default=None)
    serve.set_defaults(func=cmd_serve)

    preview = sub.add_parser("preview", help="OpenCV debug preview with ROI overlay")
    add_camera_options(preview)
    preview.set_defaults(func=cmd_preview)

    record = sub.add_parser("record", help="record a camera session for replay")
    add_camera_options(record)
    record.add_argument("--seconds", type=float, default=30.0)
    record.add_argument("--output", type=Path, default=Path("data/sessions"))
    record.set_defaults(func=cmd_record)

    replay = sub.add_parser("replay", help="replay a recorded session")
    replay.add_argument("session", type=Path)
    replay.add_argument("--recompute", action="store_true", help="re-run face/ROI tracking")
    replay.add_argument("--reference", type=Path, default=None)
    replay.add_argument("--tolerance", type=float, default=5.0)
    replay.set_defaults(func=cmd_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return int(args.func(args))
    except CaptureError as exc:
        logger.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
