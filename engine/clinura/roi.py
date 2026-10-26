from __future__ import annotations

import math

import cv2
import numpy as np

from .config import FaceConfig
from .types import ROI, FaceObservation

FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]

RIGHT_EYE = [154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173, 133, 155]
LEFT_EYE = [374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373]
RIGHT_MOUTH_CORNER = 61
LEFT_MOUTH_CORNER = 291
MOUTH_CENTER_POINTS = [61, 291, 13, 14, 0]
NOSE_TIP = 1


def _local_points(landmarks_local: np.ndarray, indices: list[int]) -> np.ndarray:
    return landmarks_local[indices, :2].copy()


def _polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.round(points).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _ellipse_mask(
    shape: tuple[int, int], center: tuple[float, float], axes: tuple[float, float]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(center[0])), int(round(center[1]))),
        (max(1, int(round(axes[0]))), max(1, int(round(axes[1])))),
        0.0,
        0.0,
        360.0,
        255,
        -1,
    )
    return mask


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _normalized_contour(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    contour = _largest_contour(mask)
    if contour is None or cv2.contourArea(contour) < 8:
        return np.empty((0, 2), dtype=np.float32)
    epsilon = max(1.0, 0.012 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float32)
    approx[:, 0] /= float(width)
    approx[:, 1] /= float(height)
    return approx


def _to_full_normalized(
    contour: np.ndarray,
    crop_width: int,
    crop_height: int,
    offset_x: int,
    offset_y: int,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    if contour.size == 0:
        return contour
    result = contour.copy()
    result[:, 0] = (result[:, 0] * crop_width + offset_x) / float(frame_width)
    result[:, 1] = (result[:, 1] * crop_height + offset_y) / float(frame_height)
    return result


def _trimmed_rgb(frame_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    pixels = frame_bgr[mask > 0]
    if pixels.size == 0:
        return (0.0, 0.0, 0.0)
    b = pixels[:, 0].astype(np.float32)
    g = pixels[:, 1].astype(np.float32)
    r = pixels[:, 2].astype(np.float32)
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    if len(luminance) >= 24:
        low = float(np.percentile(luminance, 4.0))
        high = float(np.percentile(luminance, 94.0))
        keep = (luminance >= low) & (luminance <= high)
        total = np.maximum(r + g + b, 1.0)
        chroma = np.column_stack((r / total, g / total))
        median = np.median(chroma[keep], axis=0) if keep.any() else np.median(chroma, axis=0)
        distance = np.linalg.norm(chroma - median, axis=1)
        mad = float(np.median(distance))
        if mad > 1e-6:
            keep &= distance <= max(4.0 * mad, 0.02)
        if int(keep.sum()) >= 12:
            pixels = pixels[keep]
    mean = pixels.mean(axis=0)
    return (float(mean[2]), float(mean[1]), float(mean[0]))


def _erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations <= 0:
        return mask
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(mask, kernel, iterations=iterations)


def _eye_centers_local(
    landmarks_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    right = _local_points(landmarks_local, RIGHT_EYE).mean(axis=0)
    left = _local_points(landmarks_local, LEFT_EYE).mean(axis=0)
    return right, left


def build_rois(frame_bgr: np.ndarray, landmarks: np.ndarray, config: FaceConfig) -> FaceObservation:
    frame_height, frame_width = frame_bgr.shape[:2]
    face_pts = landmarks[FACE_OVAL, :2] * np.array([frame_width, frame_height])
    face_x0 = float(face_pts[:, 0].min())
    face_x1 = float(face_pts[:, 0].max())
    face_y0 = float(face_pts[:, 1].min())
    face_y1 = float(face_pts[:, 1].max())
    face_w_full = max(1.0, face_x1 - face_x0)
    face_h_full = max(1.0, face_y1 - face_y0)

    margin_x = 0.08 * face_w_full
    margin_y = 0.08 * face_h_full
    crop_x0 = int(max(0, math.floor(face_x0 - margin_x)))
    crop_y0 = int(max(0, math.floor(face_y0 - margin_y)))
    crop_x1 = int(min(frame_width, math.ceil(face_x1 + margin_x)))
    crop_y1 = int(min(frame_height, math.ceil(face_y1 + margin_y)))
    if crop_x1 - crop_x0 < 32 or crop_y1 - crop_y0 < 32:
        return FaceObservation(
            timestamp=0.0,
            detected=True,
            landmarks=landmarks,
            frame_size=(frame_width, frame_height),
            pose_ok=False,
            reason="Face is too small",
        )

    crop = frame_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
    crop_height, crop_width = crop.shape[:2]
    local = landmarks.copy()
    local[:, 0] = landmarks[:, 0] * frame_width - crop_x0
    local[:, 1] = landmarks[:, 1] * frame_height - crop_y0
    local[:, 2] = 0.0

    face_mask = _polygon_mask((crop_height, crop_width), _local_points(local, FACE_OVAL))
    face_mask = _erode(face_mask, 1)
    ys, xs = np.nonzero(face_mask)
    if len(xs) == 0:
        return FaceObservation(
            timestamp=0.0,
            detected=True,
            landmarks=landmarks,
            frame_size=(frame_width, frame_height),
            pose_ok=False,
            reason="Face mask unavailable",
        )

    local_x0, local_x1 = int(xs.min()), int(xs.max())
    local_y0, local_y1 = int(ys.min()), int(ys.max())
    face_w = max(1.0, float(local_x1 - local_x0))
    face_h = max(1.0, float(local_y1 - local_y0))

    right_eye, left_eye = _eye_centers_local(local)
    nose = local[NOSE_TIP, :2]
    center_x = (local[234, 0] + local[454, 0]) * 0.5
    yaw_ratio = abs(float(nose[0] - center_x)) / face_w
    roll_deg = math.degrees(
        math.atan2(float(left_eye[1] - right_eye[1]), float(left_eye[0] - right_eye[0]))
    )

    pose_ok = yaw_ratio <= config.max_yaw_ratio and abs(roll_deg) <= config.max_roll_deg
    reason = None
    if not pose_ok:
        reason = "Face the camera"
    elif face_w < config.min_face_width_px:
        reason = "Move closer to the camera"

    x_margin = 0.18 * face_w
    forehead = np.zeros((crop_height, crop_width), dtype=np.uint8)
    for lo, hi in ((0.07, 0.24), (0.085, 0.22), (0.10, 0.20)):
        band = np.zeros((crop_height, crop_width), dtype=np.uint8)
        bx0 = int(local_x0 + x_margin)
        bx1 = int(local_x1 - x_margin)
        by0 = int(local_y0 + lo * face_h)
        by1 = int(local_y0 + hi * face_h)
        if bx1 > bx0 and by1 > by0:
            band[by0:by1, bx0:bx1] = 255
            candidate = cv2.bitwise_and(face_mask, band)
            candidate = _erode(candidate, 1)
            if int(np.count_nonzero(candidate)) >= 24:
                forehead = candidate
                break

    eye_radius = max(4.0, 0.095 * face_w)
    eye_cut = np.zeros((crop_height, crop_width), dtype=np.uint8)
    for eye in (right_eye, left_eye):
        cv2.circle(
            eye_cut,
            (int(round(eye[0])), int(round(eye[1]))),
            int(eye_radius),
            255,
            -1,
        )

    mouth_center = _local_points(local, MOUTH_CENTER_POINTS).mean(axis=0)
    mouth_cut = _ellipse_mask(
        (crop_height, crop_width),
        (float(mouth_center[0]), float(mouth_center[1])),
        (0.17 * face_w, 0.085 * face_h),
    )

    cheek_masks: dict[str, np.ndarray] = {}
    face_center = np.array([(local_x0 + local_x1) * 0.5, (local_y0 + local_y1) * 0.5])
    cheek_axes = (0.155 * face_w, 0.120 * face_h)
    for name, eye, mouth_corner in (
        ("right_cheek", right_eye, RIGHT_MOUTH_CORNER),
        ("left_cheek", left_eye, LEFT_MOUTH_CORNER),
    ):
        mouth_corner_local = local[mouth_corner, :2]
        center = 0.42 * eye + 0.58 * mouth_corner_local
        outward = center - face_center
        norm = float(np.linalg.norm(outward))
        if norm > 1e-3:
            center = center + outward / norm * (0.06 * face_w)
        ellipse = _ellipse_mask(
            (crop_height, crop_width),
            (float(center[0]), float(center[1])),
            cheek_axes,
        )
        candidate = cv2.bitwise_and(face_mask, ellipse)
        candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(eye_cut))
        candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(mouth_cut))
        candidate = _erode(candidate, 1)
        if int(np.count_nonzero(candidate)) < 24:
            bigger = _ellipse_mask(
                (crop_height, crop_width),
                (float(center[0]), float(center[1])),
                (cheek_axes[0] * 1.18, cheek_axes[1] * 1.18),
            )
            candidate = cv2.bitwise_and(face_mask, bigger)
            candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(eye_cut))
            candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(mouth_cut))
            candidate = _erode(candidate, 1)
        cheek_masks[name] = candidate

    masks = {"forehead": forehead, **cheek_masks}
    rois: dict[str, ROI] = {}
    for name, mask in masks.items():
        count = int(np.count_nonzero(mask))
        contour = _normalized_contour(mask, crop_width, crop_height)
        contour = _to_full_normalized(
            contour,
            crop_width,
            crop_height,
            crop_x0,
            crop_y0,
            frame_width,
            frame_height,
        )
        rois[name] = ROI(
            name=name,
            contour=contour,
            mean_rgb=_trimmed_rgb(crop, mask) if count > 0 else (0.0, 0.0, 0.0),
            pixel_count=count,
        )

    face_contour = _normalized_contour(face_mask, crop_width, crop_height)
    face_contour = _to_full_normalized(
        face_contour,
        crop_width,
        crop_height,
        crop_x0,
        crop_y0,
        frame_width,
        frame_height,
    )
    if face_contour.size == 0:
        face_contour = np.column_stack((landmarks[FACE_OVAL, 0], landmarks[FACE_OVAL, 1])).astype(
            np.float32
        )

    roi_pixels = min((r.pixel_count for r in rois.values()), default=0)
    size_score = np.clip(face_w / 220.0, 0.0, 1.0)
    roi_score = np.clip(roi_pixels / 900.0, 0.0, 1.0)
    yaw_score = (
        np.clip(1.0 - yaw_ratio / config.max_yaw_ratio, 0.0, 1.0)
        if config.max_yaw_ratio > 0
        else 1.0
    )
    roll_score = (
        np.clip(1.0 - abs(roll_deg) / config.max_roll_deg, 0.0, 1.0)
        if config.max_roll_deg > 0
        else 1.0
    )
    pose_score = float(0.6 * yaw_score + 0.4 * roll_score)

    face_pixels = crop[face_mask > 0]
    if face_pixels.size:
        luminance = (
            0.114 * face_pixels[:, 0].astype(np.float32)
            + 0.587 * face_pixels[:, 1].astype(np.float32)
            + 0.299 * face_pixels[:, 2].astype(np.float32)
        )
        mean_luma = float(luminance.mean())
    else:
        mean_luma = 0.0
    if mean_luma < 55.0:
        brightness_score = np.clip(mean_luma / 55.0, 0.0, 1.0) * 0.7
        if reason is None:
            reason = "Lighting is too low"
    elif mean_luma > 215.0:
        brightness_score = np.clip((255.0 - mean_luma) / 40.0, 0.0, 1.0) * 0.7
        if reason is None:
            reason = "Face is overexposed"
    else:
        brightness_score = 1.0

    quality = float(
        np.clip(
            0.34 * size_score + 0.26 * roi_score + 0.22 * pose_score + 0.18 * brightness_score,
            0.0,
            1.0,
        )
    )
    if not pose_ok:
        quality *= 0.35

    return FaceObservation(
        timestamp=0.0,
        detected=True,
        landmarks=landmarks,
        bbox=(
            crop_x0 + local_x0,
            crop_y0 + local_y0,
            int(face_w),
            int(face_h),
        ),
        rois=rois,
        face_contour=face_contour,
        quality=quality,
        pose_ok=pose_ok,
        reason=reason,
        frame_size=(frame_width, frame_height),
        face_width_px=face_w,
        yaw_ratio=yaw_ratio,
        roll_deg=roll_deg,
        mean_luma=mean_luma,
    )
