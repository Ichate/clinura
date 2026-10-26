from __future__ import annotations

import math
from collections import deque

import numpy as np


class EyeActivityTracker:
    def __init__(self) -> None:
        self._blinks: deque[float] = deque()
        self._closed = False
        self._openness = 1.0
        self._started_at: float | None = None

    def update(self, blink_left: float, blink_right: float, timestamp: float) -> dict[str, object]:
        if self._started_at is None:
            self._started_at = timestamp
        blink = max(float(blink_left), float(blink_right))
        self._openness = 1.0 - blink
        if blink > 0.55 and not self._closed:
            self._blinks.append(timestamp)
            self._closed = True
        elif blink < 0.3:
            self._closed = False
        cutoff = timestamp - 60.0
        while self._blinks and self._blinks[0] < cutoff:
            self._blinks.popleft()
        elapsed = min(60.0, max(0.0, timestamp - self._started_at))
        rate = None
        if elapsed >= 5.0:
            rate = float(len(self._blinks) * 60.0 / max(elapsed, 1.0))
        return {
            "blink_rate": None if rate is None else round(rate, 1),
            "blinks": int(len(self._blinks)),
            "openness": round(float(self._openness), 3),
        }

    def reset(self) -> None:
        self._blinks.clear()
        self._closed = False
        self._openness = 1.0
        self._started_at = None


class MotionTracker:
    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self._history: deque[tuple[float, float, float, float, float, float]] = deque()

    def update(
        self,
        timestamp: float,
        yaw_ratio: float,
        roll_deg: float,
        bbox: tuple[int, int, int, int] | None,
    ) -> dict[str, object]:
        if bbox is None:
            self._history.clear()
            return {
                "score": 0.0,
                "motion_px_s": None,
                "yaw_ratio": round(float(yaw_ratio), 3),
                "roll_deg": round(float(roll_deg), 1),
            }
        x, y, w, h = bbox
        cx = x + w * 0.5
        cy = y + h * 0.5
        self._history.append((timestamp, cx, cy, float(w), float(yaw_ratio), float(roll_deg)))
        cutoff = timestamp - self.window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        history = list(self._history)
        motion = 0.0
        scale_change = 0.0
        if len(history) >= 2:
            span = max(1e-6, history[-1][0] - history[0][0])
            distance = 0.0
            for first, second in zip(history, history[1:], strict=False):
                distance += math.hypot(second[1] - first[1], second[2] - first[2])
            motion = distance / span
            widths = np.asarray([item[3] for item in history], dtype=np.float64)
            mean_width = float(np.mean(widths))
            if mean_width > 1e-6:
                scale_change = float(np.std(widths) / mean_width)

        motion_score = min(1.0, motion / 60.0 + scale_change * 2.0)
        pose_penalty = max(
            min(1.0, abs(float(yaw_ratio)) / 0.3),
            min(1.0, abs(float(roll_deg)) / 25.0),
        )
        stability = max(0.0, min(1.0, 1.0 - 0.55 * motion_score - 0.45 * pose_penalty))
        return {
            "score": round(stability * 100.0, 1),
            "motion_px_s": round(float(motion), 1),
            "yaw_ratio": round(float(yaw_ratio), 3),
            "roll_deg": round(float(roll_deg), 1),
        }

    def reset(self) -> None:
        self._history.clear()
