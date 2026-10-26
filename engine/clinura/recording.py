from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import CaptureConfig, FaceConfig, RecordConfig, SignalConfig
from .types import FaceObservation

logger = logging.getLogger(__name__)


@dataclass
class RecordedFrame:
    frame_index: int
    timestamp: float
    wall_time: float
    face_detected: bool
    quality: float
    pose_ok: bool
    reason: str | None
    bbox: list[int] | None
    face_width_px: float
    roi_pixels: int
    rois: dict[str, dict[str, object]]


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


class SessionRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        capture: CaptureConfig | None = None,
        face: FaceConfig | None = None,
        signal: SignalConfig | None = None,
        name: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if name is not None:
            self.output_dir = self.output_dir / name
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps > 0 else 30.0
        self.frame_size = frame_size
        self.capture = capture or CaptureConfig()
        self.face = face or FaceConfig()
        self.signal = signal or SignalConfig()
        self._frames_path = self.output_dir / "frames.jsonl"
        self._frames_file = self._frames_path.open("w", encoding="utf-8")
        self._video_path = self.output_dir / f"video{RecordConfig().video_ext}"
        self._writer = cv2.VideoWriter(
            str(self._video_path),
            cv2.VideoWriter_fourcc(*RecordConfig().video_fourcc),
            self.fps,
            frame_size,
        )
        self._codec = RecordConfig().video_fourcc
        if not self._writer.isOpened():
            logger.warning("FFV1 writer unavailable; falling back to MJPG")
            self._writer = cv2.VideoWriter(
                str(self._video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self.fps,
                frame_size,
            )
            self._codec = "MJPG"
        if not self._writer.isOpened():
            raise RuntimeError(f"Unable to open video writer at {self._video_path}")
        self.frame_count = 0
        self._started_at = time.time()
        self._started_wall = datetime.now(UTC).isoformat()
        self._closed = False

    def write(
        self,
        frame_bgr: np.ndarray,
        observation: FaceObservation,
        timestamp: float,
        wall_time: float | None = None,
    ) -> RecordedFrame:
        if self._closed:
            raise RuntimeError("recorder is closed")
        if frame_bgr.shape[1] != self.frame_size[0] or frame_bgr.shape[0] != self.frame_size[1]:
            frame_bgr = cv2.resize(frame_bgr, self.frame_size, interpolation=cv2.INTER_AREA)
        wall = float(wall_time if wall_time is not None else time.time())
        bbox = list(observation.bbox) if observation.bbox is not None else None
        rois: dict[str, dict[str, object]] = {}
        for name, roi in observation.rois.items():
            rois[name] = {
                "rgb": [_round(v, 3) for v in roi.mean_rgb],
                "pixels": int(roi.pixel_count),
            }
        record = RecordedFrame(
            frame_index=self.frame_count,
            timestamp=float(timestamp),
            wall_time=wall,
            face_detected=bool(observation.detected),
            quality=_round(observation.quality),
            pose_ok=bool(observation.pose_ok),
            reason=observation.reason,
            bbox=bbox,
            face_width_px=_round(observation.face_width_px, 2),
            roi_pixels=int(observation.roi_pixels),
            rois=rois,
        )
        self._writer.write(frame_bgr)
        self._frames_file.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self._frames_file.flush()
        return record

    def close(self) -> Path:
        if self._closed:
            return self.output_dir
        self._closed = True
        self._frames_file.flush()
        self._frames_file.close()
        self._writer.release()
        duration = time.time() - self._started_at
        meta = {
            "version": __version__,
            "started_at": self._started_wall,
            "ended_at": datetime.now(UTC).isoformat(),
            "duration_s": round(duration, 3),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "measured_fps": round(self.frame_count / duration, 3) if duration > 0 else None,
            "frame_size": list(self.frame_size),
            "video": self._video_path.name,
            "codec": self._codec,
            "capture": asdict(self.capture),
            "face": {
                "smoothing": self.face.smoothing,
                "min_face_width_px": self.face.min_face_width_px,
                "max_yaw_ratio": self.face.max_yaw_ratio,
                "max_roll_deg": self.face.max_roll_deg,
            },
            "signal": asdict(self.signal),
        }
        (self.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Recorded %d frames to %s", self.frame_count, self.output_dir)
        return self.output_dir


class RecordedSession:
    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        meta_path = self.session_dir / "meta.json"
        frames_path = self.session_dir / "frames.jsonl"
        if not meta_path.exists() or not frames_path.exists():
            raise FileNotFoundError(f"Not a Clinura session: {self.session_dir}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.records: list[RecordedFrame] = []
        with frames_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.records.append(RecordedFrame(**json.loads(line)))
        video_name = self.meta.get("video", "video.avi")
        self.video_path = self.session_dir / video_name
        self._cap = cv2.VideoCapture(str(self.video_path)) if self.video_path.exists() else None

    def iter_frames(self) -> Iterator[tuple[RecordedFrame, np.ndarray | None]]:
        if self._cap is None:
            for record in self.records:
                yield record, None
            return
        for record in self.records:
            ok, frame = self._cap.read()
            if not ok:
                yield record, None
                continue
            yield record, frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def sample_rgb(self, record: RecordedFrame) -> tuple[float, float, float] | None:
        rois = record.rois
        if not rois:
            return None
        total = 0
        accum = np.zeros(3, dtype=np.float64)
        for roi in rois.values():
            pixels = int(roi.get("pixels", 0))
            rgb = roi.get("rgb")
            if pixels <= 0 or not isinstance(rgb, list) or len(rgb) != 3:
                continue
            accum += np.asarray(rgb, dtype=np.float64) * pixels
            total += pixels
        if total <= 0:
            return None
        mean = accum / total
        return (float(mean[0]), float(mean[1]), float(mean[2]))
