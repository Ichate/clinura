from __future__ import annotations

import logging

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import FaceConfig
from .roi import build_rois
from .types import FaceObservation

logger = logging.getLogger(__name__)


class LandmarkSmoother:
    def __init__(self, alpha: float = 0.55) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._prev: np.ndarray | None = None

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        current = landmarks.astype(np.float32, copy=True)
        if self._prev is None or self._prev.shape != current.shape:
            self._prev = current
        else:
            self._prev = self.alpha * current + (1.0 - self.alpha) * self._prev
        return self._prev.copy()

    def reset(self) -> None:
        self._prev = None


class FaceTracker:
    def __init__(self, config: FaceConfig | None = None) -> None:
        self.config = config or FaceConfig()
        self._last_ts_ms = -1
        self._smoother = LandmarkSmoother(self.config.smoothing)
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._video = self._create_landmarker(vision.RunningMode.VIDEO)
        self._image = (
            self._create_landmarker(vision.RunningMode.IMAGE) if self.config.fallback_crop else None
        )

    def _create_landmarker(self, mode: vision.RunningMode) -> vision.FaceLandmarker:
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.config.model_path)),
            running_mode=mode,
            num_faces=self.config.num_faces,
            min_face_detection_confidence=self.config.min_face_detection_confidence,
            min_face_presence_confidence=self.config.min_face_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_s: float) -> FaceObservation:
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

        result = self._video.detect_for_video(image, ts_ms)
        landmarks = self._extract_landmarks(result)

        if landmarks is None and self._image is not None:
            landmarks = self._detect_with_crop(frame_bgr)

        if landmarks is None:
            self._smoother.reset()
            self._last_bbox = None
            return FaceObservation(
                timestamp=timestamp_s,
                detected=False,
                reason="No face detected",
                frame_size=(width, height),
            )

        smoothed = self._smoother.update(landmarks)
        observation = build_rois(frame_bgr, smoothed, self.config)
        blink_left, blink_right = self._extract_blinks(result)
        observation.blink_left = blink_left
        observation.blink_right = blink_right
        observation.timestamp = float(timestamp_s)
        observation.frame_size = (width, height)
        if observation.bbox is not None:
            self._last_bbox = observation.bbox
        return observation

    def _extract_blinks(self, result: vision.FaceLandmarkerResult) -> tuple[float, float]:
        left = 0.0
        right = 0.0
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                if category.category_name == "eyeBlinkLeft":
                    left = float(category.score)
                elif category.category_name == "eyeBlinkRight":
                    right = float(category.score)
        return left, right

    def _extract_landmarks(self, result: vision.FaceLandmarkerResult) -> np.ndarray | None:
        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        points = np.asarray([[p.x, p.y, p.z] for p in face], dtype=np.float32)
        if points.shape[0] < 468:
            return None
        return points

    def _detect_with_crop(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        if self._image is None:
            return None
        height, width = frame_bgr.shape[:2]
        if self._last_bbox is not None:
            bx, by, bw, bh = self._last_bbox
            margin_x = 0.75 * bw
            margin_y = 0.75 * bh
            crop_x0 = max(0, int(bx - margin_x))
            crop_y0 = max(0, int(by - margin_y))
            crop_x1 = min(width, int(bx + bw + margin_x))
            crop_y1 = min(height, int(by + bh + margin_y))
        else:
            crop_w = int(self.config.fallback_crop_width_ratio * width)
            crop_h = int(self.config.fallback_crop_height_ratio * height)
            crop_x0 = max(0, (width - crop_w) // 2)
            crop_y0 = max(0, (height - crop_h) // 2)
            crop_x1 = crop_x0 + crop_w
            crop_y1 = crop_y0 + crop_h

        if crop_x1 - crop_x0 < 32 or crop_y1 - crop_y0 < 32:
            return None

        crop = frame_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
        scale = min(1.0, self.config.fallback_input_width / max(1, crop.shape[1]))
        if scale < 1.0:
            crop = cv2.resize(
                crop,
                (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        elif crop.shape[1] < self.config.fallback_input_width:
            scale_up = self.config.fallback_input_width / max(1, crop.shape[1])
            crop = cv2.resize(
                crop,
                (int(crop.shape[1] * scale_up), int(crop.shape[0] * scale_up)),
                interpolation=cv2.INTER_CUBIC,
            )
            scale = scale_up

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._image.detect(image)
        points = self._extract_landmarks(result)
        if points is None:
            return None

        crop_w = crop_x1 - crop_x0
        crop_h = crop_y1 - crop_y0
        mapped = points.copy()
        mapped[:, 0] = (crop_x0 + points[:, 0] * crop_w) / float(width)
        mapped[:, 1] = (crop_y0 + points[:, 1] * crop_h) / float(height)
        mapped[:, 2] = points[:, 2] * (crop_w / float(width))
        return mapped

    def close(self) -> None:
        self._video.close()
        if self._image is not None:
            self._image.close()
