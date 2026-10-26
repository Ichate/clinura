from __future__ import annotations

import cv2
import numpy as np

from clinura.config import FaceConfig
from clinura.roi import (
    FACE_OVAL,
    LEFT_EYE,
    LEFT_MOUTH_CORNER,
    NOSE_TIP,
    RIGHT_EYE,
    RIGHT_MOUTH_CORNER,
    build_rois,
)


def _make_landmarks() -> np.ndarray:
    landmarks = np.full((478, 3), 0.5, dtype=np.float32)
    cx, cy = 0.5, 0.52
    rx, ry = 0.18, 0.28
    for index, landmark_index in enumerate(FACE_OVAL):
        theta = 2.0 * np.pi * index / len(FACE_OVAL)
        landmarks[landmark_index] = [
            cx + rx * np.sin(theta),
            cy - ry * np.cos(theta),
            0.0,
        ]
    landmarks[234] = [cx - rx, cy, 0.0]
    landmarks[454] = [cx + rx, cy, 0.0]
    landmarks[NOSE_TIP] = [cx, cy, 0.0]

    for indices, center in ((RIGHT_EYE, (0.40, 0.43)), (LEFT_EYE, (0.60, 0.43))):
        for index, landmark_index in enumerate(indices):
            theta = 2.0 * np.pi * index / len(indices)
            landmarks[landmark_index] = [
                center[0] + 0.028 * np.cos(theta),
                center[1] + 0.016 * np.sin(theta),
                0.0,
            ]

    landmarks[RIGHT_MOUTH_CORNER] = [0.43, 0.63, 0.0]
    landmarks[LEFT_MOUTH_CORNER] = [0.57, 0.63, 0.0]
    landmarks[0] = [0.50, 0.60, 0.0]
    landmarks[13] = [0.50, 0.615, 0.0]
    landmarks[14] = [0.50, 0.635, 0.0]
    return landmarks


def test_build_rois_on_synthetic_face() -> None:
    width, height = 640, 480
    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.ellipse(
        frame,
        (int(0.5 * width), int(0.52 * height)),
        (int(0.18 * width), int(0.28 * height)),
        0.0,
        0.0,
        360.0,
        (90, 120, 160),
        -1,
    )
    landmarks = _make_landmarks()
    observation = build_rois(frame, landmarks, FaceConfig())

    assert observation.detected
    assert observation.bbox is not None
    assert observation.face_width_px > 100
    assert set(observation.rois) == {"forehead", "right_cheek", "left_cheek"}
    for roi in observation.rois.values():
        assert roi.pixel_count > 20
        assert roi.contour.shape[1] == 2
        assert roi.mean_rgb[0] > roi.mean_rgb[1] > roi.mean_rgb[2]
    assert observation.pose_ok
    assert not observation.reason
