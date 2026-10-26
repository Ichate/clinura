from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import FaceConfig, SignalConfig
from .face import FaceTracker
from .pipeline import SignalPipeline
from .recording import RecordedSession
from .types import AnalysisResult

logger = logging.getLogger(__name__)


def _result_dict(result: AnalysisResult) -> dict[str, object]:
    def value(number: float | None, digits: int = 3) -> float | None:
        return None if number is None else round(float(number), digits)

    return {
        "timestamp": round(float(result.timestamp), 4),
        "wall_time": None if result.wall_time is None else round(float(result.wall_time), 3),
        "status": result.status,
        "message": result.message,
        "bpm": value(result.bpm),
        "provisional_bpm": value(result.provisional_bpm),
        "snr_db": value(result.snr_db),
        "rmssd_ms": value(result.rmssd_ms),
        "hrv_ready": result.hrv_ready,
        "hrv_progress": value(result.hrv_progress),
        "confidence": value(result.confidence),
        "fs": value(result.fs),
        "window_s": value(result.window_s),
        "sample_count": result.sample_count,
        "ibi_count": result.ibi_count,
        "analysis_ms": value(result.analysis_ms, 2),
    }


def replay_session(
    session_dir: str | Path,
    recompute: bool = False,
    face_config: FaceConfig | None = None,
    signal_config: SignalConfig | None = None,
    write_analysis: bool = True,
) -> list[AnalysisResult]:
    session = RecordedSession(session_dir)
    config = signal_config or SignalConfig()
    pipeline = SignalPipeline(config)
    results: list[AnalysisResult] = []
    last_analysis = 0.0
    last_timestamp = 0.0
    tracker = FaceTracker(face_config or FaceConfig()) if recompute else None
    record_times = np.asarray([r.timestamp for r in session.records], dtype=np.float64)
    record_walls = np.asarray([r.wall_time for r in session.records], dtype=np.float64)

    def wall_time_for(timestamp: float) -> float | None:
        if record_times.size == 0:
            return None
        index = int(np.argmin(np.abs(record_times - timestamp)))
        return float(record_walls[index])

    try:
        for record, frame in session.iter_frames():
            last_timestamp = float(record.timestamp)
            if recompute:
                if frame is None:
                    continue
                observation = tracker.detect(frame, record.timestamp)
                pipeline.set_face_status(
                    observation.detected,
                    observation.quality,
                    observation.reason,
                    record.timestamp,
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
                        record.timestamp,
                        rgb,
                        observation.quality,
                        observation.face_width_px,
                        observation.roi_pixels,
                        roi_rgb=roi_rgb,
                        roi_pixels_by_name=roi_pixels_by_name,
                    )
            else:
                pipeline.set_face_status(
                    record.face_detected,
                    record.quality,
                    record.reason,
                    record.timestamp,
                    pose_ok=record.pose_ok,
                )
                rgb = session.sample_rgb(record)
                if (
                    record.face_detected
                    and record.pose_ok
                    and rgb is not None
                    and record.quality >= 0.08
                ):
                    roi_rgb = {
                        name: tuple(roi["rgb"])
                        for name, roi in record.rois.items()
                        if roi.get("pixels", 0) > 0 and isinstance(roi.get("rgb"), list)
                    }
                    roi_pixels_by_name = {
                        name: int(roi.get("pixels", 0)) for name, roi in record.rois.items()
                    }
                    pipeline.add_sample(
                        record.timestamp,
                        rgb,
                        record.quality,
                        record.face_width_px,
                        record.roi_pixels,
                        roi_rgb=roi_rgb,
                        roi_pixels_by_name=roi_pixels_by_name,
                    )
            if last_timestamp - last_analysis >= config.analysis_interval_s:
                result = pipeline.analyze(last_timestamp)
                result.wall_time = wall_time_for(result.timestamp)
                results.append(result)
                last_analysis = last_timestamp
    finally:
        if tracker is not None:
            tracker.close()
        session.close()

    final = pipeline.analyze(last_timestamp, force=True)
    final.wall_time = wall_time_for(final.timestamp)
    if not results or results[-1].timestamp != final.timestamp:
        results.append(final)

    if write_analysis:
        output = session.session_dir / "analysis.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(_result_dict(result), separators=(",", ":")) + "\n")
        logger.info("Wrote %s", output)

    if final.bpm is not None:
        print(
            f"replay: {final.status} bpm={final.bpm:.1f} "
            f"snr={final.snr_db:.1f}dB rmssd={final.rmssd_ms} "
            f"confidence={final.confidence:.2f}"
        )
    else:
        print(f"replay: {final.status} ({final.message})")
    return results


def compare_to_reference(
    analysis_path: str | Path,
    reference_path: str | Path,
    tolerance_bpm: float = 5.0,
) -> dict[str, float]:
    analysis: list[dict[str, object]] = []
    with Path(analysis_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                analysis.append(json.loads(line))
    reference: list[tuple[float, float]] = []
    with Path(reference_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            reference.append((float(item["timestamp"]), float(item["bpm"])))

    errors: list[float] = []
    for item in analysis:
        bpm = item.get("bpm")
        if bpm is None:
            continue
        timestamp = float(item.get("wall_time") or item["timestamp"])
        nearest = min(reference, key=lambda pair: abs(pair[0] - timestamp), default=None)
        if nearest is None:
            continue
        errors.append(float(bpm) - nearest[1])

    if not errors:
        return {
            "n": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "within_tolerance_pct": float("nan"),
        }
    array = np.asarray(errors, dtype=np.float64)
    return {
        "n": float(len(array)),
        "mae": float(np.mean(np.abs(array))),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "within_tolerance_pct": float(np.mean(np.abs(array) <= tolerance_bpm) * 100.0),
    }
