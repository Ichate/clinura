from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .roi import (
    LEFT_MOUTH_CORNER,
    RIGHT_MOUTH_CORNER,
    _ellipse_mask,
    _erode,
    _normalized_contour,
    _polygon_mask,
    _trimmed_rgb,
)
from .types import ROI, FaceObservation

INNER_MOUTH = [
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
]
UPPER_INNER_LIP = 13
LOWER_INNER_LIP = 14


@dataclass
class ExperimentalObservation:
    mode: str
    active: bool
    roi: ROI | None = None
    quality: float = 0.0
    reason: str | None = None
    handedness: str | None = None


def build_tongue_roi(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray,
    face: FaceObservation,
) -> ExperimentalObservation:
    height, width = frame_bgr.shape[:2]
    face_width = max(1.0, face.face_width_px)
    mouth_open = float(
        np.linalg.norm(landmarks[UPPER_INNER_LIP, :2] - landmarks[LOWER_INNER_LIP, :2])
        * height
        / face_width
    )
    if mouth_open < 0.035:
        return ExperimentalObservation(
            mode="tongue", active=False, reason="Open your mouth and show your tongue"
        )

    mouth_points = landmarks[INNER_MOUTH, :2] * np.array([width, height])
    mouth_mask = _polygon_mask((height, width), mouth_points)
    mouth_mask = _erode(mouth_mask, 2)
    if int(np.count_nonzero(mouth_mask)) < 40:
        return ExperimentalObservation(
            mode="tongue", active=False, reason="Open wider to expose your tongue"
        )

    mouth_corners = landmarks[[RIGHT_MOUTH_CORNER, LEFT_MOUTH_CORNER], :2] * np.array(
        [width, height]
    )
    mouth_center = landmarks[[UPPER_INNER_LIP, LOWER_INNER_LIP], :2].mean(axis=0) * np.array(
        [width, height]
    )
    mouth_width = float(np.linalg.norm(mouth_corners[1] - mouth_corners[0]))
    face_height = float(face.bbox[3]) if face.bbox is not None else face_width * 1.3
    lower_center = (
        float(mouth_center[0]),
        float(mouth_center[1] + 0.055 * face_height),
    )
    lower_region = _ellipse_mask(
        (height, width),
        lower_center,
        (max(24.0, 0.55 * mouth_width), max(24.0, 0.18 * face_height)),
    )
    region = cv2.bitwise_or(mouth_mask, lower_region)
    region = _erode(region, 1)

    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.int16)
    cr = ycrcb[:, :, 1].astype(np.int16)
    cb = ycrcb[:, :, 2].astype(np.int16)
    color = ((cr >= 138) & ((cr - cb) >= 12) & (y >= 30)).astype(np.uint8) * 255
    tongue = cv2.bitwise_and(color, region)
    tongue = cv2.morphologyEx(tongue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    tongue = cv2.morphologyEx(tongue, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    tongue = _erode(tongue, 1)

    contours, _ = cv2.findContours(tongue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ExperimentalObservation(
            mode="tongue", active=False, reason="Show your tongue in the mouth opening"
        )
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 35:
        return ExperimentalObservation(
            mode="tongue", active=False, reason="Show more of your tongue"
        )

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    mask = _erode(mask, 1)
    pixels = int(np.count_nonzero(mask))
    if pixels < 30:
        return ExperimentalObservation(
            mode="tongue", active=False, reason="Show more of your tongue"
        )

    mean_rgb = _trimmed_rgb(frame_bgr, mask)
    contour = _normalized_contour(mask, width, height)
    luma = 0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2]
    open_score = np.clip((mouth_open - 0.03) / 0.09, 0.0, 1.0)
    area_score = np.clip(pixels / 700.0, 0.0, 1.0)
    if luma < 45:
        brightness_score = np.clip(luma / 45.0, 0.0, 1.0) * 0.7
    elif luma > 225:
        brightness_score = np.clip((255.0 - luma) / 30.0, 0.0, 1.0) * 0.7
    else:
        brightness_score = 1.0
    quality = float(
        np.clip(
            0.4 * open_score + 0.4 * area_score + 0.2 * brightness_score
            if face.pose_ok
            else 0.15 * (0.4 * open_score + 0.4 * area_score + 0.2 * brightness_score),
            0.0,
            1.0,
        )
    )
    return ExperimentalObservation(
        mode="tongue",
        active=True,
        roi=ROI(name="tongue", contour=contour, mean_rgb=mean_rgb, pixel_count=pixels),
        quality=quality,
        reason=None,
    )
