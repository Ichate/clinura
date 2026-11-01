from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import PoseConfig


@dataclass
class PoseObservation:
    detected: bool
    timestamp: float
    image: np.ndarray | None = None  # (33, 3): normalized x, y, visibility
    world: np.ndarray | None = None  # (33, 3): metres, hip-centred, y down


class PoseTracker:
    """MediaPipe Pose Landmarker (full model) in VIDEO mode."""

    def __init__(self, config: PoseConfig | None = None) -> None:
        self.config = config or PoseConfig()
        self._last_ts_ms = -1
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.config.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_detection_confidence,
            min_pose_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_s: float) -> PoseObservation:
        height, width = frame_bgr.shape[:2]
        image = frame_bgr
        if width > self.config.detection_width:
            scale = self.config.detection_width / float(width)
            image = cv2.resize(
                frame_bgr,
                (self.config.detection_width, max(2, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ts_ms = max(int(timestamp_s * 1000.0), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
        )
        if not result.pose_landmarks:
            return PoseObservation(detected=False, timestamp=timestamp_s)
        points = result.pose_landmarks[0]
        image_pts = np.asarray(
            [[p.x, p.y, float(p.visibility or 0.0)] for p in points], dtype=np.float32
        )
        world = None
        if result.pose_world_landmarks:
            world = np.asarray(
                [[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]], dtype=np.float32
            )
        return PoseObservation(detected=True, timestamp=timestamp_s, image=image_pts, world=world)

    def close(self) -> None:
        self._landmarker.close()
