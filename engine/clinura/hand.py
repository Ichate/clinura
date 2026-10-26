from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import HandConfig
from .experimental import ExperimentalObservation
from .roi import _erode, _normalized_contour, _trimmed_rgb
from .types import ROI

PALM_INDICES = [0, 1, 2, 5, 9, 13, 17]


class HandTracker:
    def __init__(self, config: HandConfig | None = None) -> None:
        self.config = config or HandConfig()
        self._last_ts_ms = -1
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.config.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_hand_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_s: float) -> ExperimentalObservation:
        height, width = frame_bgr.shape[:2]
        detection = frame_bgr
        if width > self.config.detection_width:
            scale = self.config.detection_width / float(width)
            detection = cv2.resize(
                frame_bgr,
                (self.config.detection_width, max(2, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(detection, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = max(int(timestamp_s * 1000.0), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.hand_landmarks:
            return ExperimentalObservation(
                mode="palm", active=False, reason="Show your palm to the camera"
            )

        landmarks = np.asarray(
            [[point.x, point.y, point.z] for point in result.hand_landmarks[0]],
            dtype=np.float32,
        )
        handedness = None
        confidence = 0.0
        if result.handedness:
            categories = result.handedness[0]
            if categories:
                handedness = str(categories[0].category_name)
                confidence = float(categories[0].score)

        points = landmarks[PALM_INDICES, :2] * np.array([width, height])
        points = points.astype(np.int32)
        hull = cv2.convexHull(points)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        mask = _erode(mask, 2)
        pixels = int(np.count_nonzero(mask))
        if pixels < 250:
            return ExperimentalObservation(
                mode="palm", active=False, reason="Move your palm closer", handedness=handedness
            )

        x, y, w, h = cv2.boundingRect(hull)
        size = max(1.0, float(w))
        mean_rgb = _trimmed_rgb(frame_bgr, mask)
        contour = _normalized_contour(mask, width, height)
        luma = 0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2]
        size_score = np.clip(size / 260.0, 0.0, 1.0)
        area_score = np.clip(pixels / 8000.0, 0.0, 1.0)
        if luma < 45:
            brightness_score = np.clip(luma / 45.0, 0.0, 1.0) * 0.7
        elif luma > 225:
            brightness_score = np.clip((255.0 - luma) / 30.0, 0.0, 1.0) * 0.7
        else:
            brightness_score = 1.0
        quality = float(
            np.clip(
                0.35 * size_score + 0.30 * area_score + 0.20 * confidence + 0.15 * brightness_score,
                0.0,
                1.0,
            )
        )
        return ExperimentalObservation(
            mode="palm",
            active=True,
            roi=ROI(name="palm", contour=contour, mean_rgb=mean_rgb, pixel_count=pixels),
            quality=quality,
            reason=None,
            handedness=handedness,
        )

    def close(self) -> None:
        self._landmarker.close()
