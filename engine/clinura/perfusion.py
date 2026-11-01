"""Webcam "thermal" view: blood-perfusion and flushing maps.

A webcam cannot measure temperature. What it can measure is *blood flow in the skin*,
which is what makes skin warm: each small patch of the face pulses with the heartbeat,
and the strength of that pulse (relative to the patch's noise) maps perfusion. Shown
with a thermal colour scale this gives a heat-map-like view that is physiologically
real, plus:

* left/right perfusion asymmetry (e.g. one-sided flushing or reduced flow);
* redness (CIE a*) change against a baseline taken when the mode starts: flushing after
  exercise, heat or embarrassment raises it, cold/pallor lowers it.
"""

from __future__ import annotations

import base64
import time
from collections import deque

import cv2
import numpy as np

GRID = 24
WINDOW_S = 8.0


class PerfusionMapper:
    def __init__(self) -> None:
        self._frames: deque[tuple[float, np.ndarray]] = deque()
        self._redness: deque[tuple[float, float]] = deque(maxlen=900)
        self._baseline_redness: float | None = None
        self._map: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._last_map_t = 0.0
        self._box: tuple[int, int, int, int] | None = None
        self._image: str | None = None
        self._stats: dict[str, object] = {}
        self._history: deque[tuple[float, float | None, float | None]] = deque(maxlen=600)
        self._last_hist = 0.0

    def reset(self) -> None:
        self.__init__()

    def update(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int] | None,
        face_mask_fn,
        bpm: float | None,
        t: float,
    ) -> None:
        if box is None:
            self._frames.clear()
            self._box = None
            return
        x, y, w, h = box
        if self._box is None:
            self._box = box
        else:  # smooth the crop so blocks stay on the same skin
            self._box = tuple(
                int(round(0.8 * a + 0.2 * b)) for a, b in zip(self._box, box, strict=True)
            )
        x, y, w, h = self._box
        crop = frame[max(0, y) : y + h, max(0, x) : x + w]
        if crop.size == 0 or min(crop.shape[:2]) < GRID:
            return
        small = cv2.resize(crop, (GRID, GRID), interpolation=cv2.INTER_AREA).astype(np.float32)
        self._frames.append((t, small))
        while self._frames and self._frames[0][0] < t - WINDOW_S:
            self._frames.popleft()
        if self._mask is None or t - self._last_map_t > 5.0:
            self._mask = face_mask_fn(self._box, GRID)
        lab = cv2.cvtColor(np.clip(small, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        skin = self._mask if self._mask is not None else np.ones((GRID, GRID), bool)
        redness = float(np.mean(lab[:, :, 1][skin]) - 128.0)
        self._redness.append((t, redness))
        if self._baseline_redness is None and len(self._redness) >= 60:
            self._baseline_redness = float(np.median([r for _, r in self._redness]))
        if bpm is not None and t - self._last_map_t >= 0.5 and len(self._frames) >= 90:
            self._compute(bpm, t)

    def _compute(self, bpm: float, t: float) -> None:
        self._last_map_t = t
        ts = np.asarray([f[0] for f in self._frames])
        stack = np.stack([f[1] for f in self._frames])  # (n, G, G, 3) BGR
        fs = (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)
        grid = np.linspace(ts[0], ts[-1], len(ts))
        flat = stack.reshape(len(ts), -1, 3)
        # Per-block CHROM-like pulse: normalise each channel, combine to cancel motion.
        norm = flat / np.maximum(flat.mean(axis=0, keepdims=True), 1e-6)
        b, g, r = norm[:, :, 0], norm[:, :, 1], norm[:, :, 2]
        xs = 3 * r - 2 * g
        ys = 1.5 * r + g - 1.5 * b
        alpha = xs.std(axis=0) / np.maximum(ys.std(axis=0), 1e-9)
        pulse = xs - alpha * ys
        pulse = pulse - pulse.mean(axis=0)
        pulse = np.asarray([np.interp(grid, ts, pulse[:, k]) for k in range(pulse.shape[1])]).T
        window = np.hanning(len(grid))[:, None]
        spec = np.abs(np.fft.rfft(pulse * window, n=1024, axis=0)) ** 2
        freqs = np.fft.rfftfreq(1024, 1 / fs)
        f0 = bpm / 60.0
        sig = (np.abs(freqs - f0) <= 0.12)[:, None]
        band = ((freqs >= 0.65) & (freqs <= 4.0))[:, None]
        snr = (spec * sig).sum(axis=0) / np.maximum(
            (spec * band).sum(axis=0) - (spec * sig).sum(axis=0), 1e-12
        )
        perf = np.sqrt((spec * sig).sum(axis=0)) / np.maximum(flat[:, :, 1].mean(axis=0), 1e-6)
        score = perf * np.clip(snr / (snr + 0.3), 0, 1)
        score = score.reshape(GRID, GRID)
        if self._mask is not None:
            score = np.where(self._mask, score, np.nan)
        valid = np.isfinite(score)
        if valid.sum() < 20:
            return
        scale = np.nanpercentile(score, 95) or 1.0
        current = np.clip(np.nan_to_num(score / scale, nan=0.0), 0, 1.2)
        self._map = current if self._map is None else 0.6 * self._map + 0.4 * current
        left = np.nanmean(score[:, : GRID // 2])
        right = np.nanmean(score[:, GRID // 2 :])
        asym = float(abs(left - right) / max((left + right) / 2, 1e-12) * 100.0)
        redness = float(np.median([r for _, r in list(self._redness)[-30:]]))
        flush = None if self._baseline_redness is None else redness - self._baseline_redness
        self._stats = {
            "perfusion_index": round(float(np.nanmean(score) / scale * 100.0), 1),
            "asymmetry_pct": round(asym, 1),
            "warmer_side": None if asym < 15 else ("left" if left > right else "right"),
            "redness": round(redness, 2),
            "flush_delta": None if flush is None else round(flush, 2),
            "coverage_pct": round(100.0 * valid.mean(), 1),
        }
        if t - self._last_hist >= 1.0:
            self._last_hist = t
            self._history.append((time.time(), self._stats["perfusion_index"], flush))
        heat = (np.clip(self._map / 1.2, 0, 1) * 255).astype(np.uint8)
        heat = cv2.resize(heat, (240, 240), interpolation=cv2.INTER_CUBIC)
        color = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
        mask_big = cv2.resize(
            (self._mask if self._mask is not None else np.ones((GRID, GRID), bool)).astype(np.uint8)
            * 255,
            (240, 240),
        )
        color[mask_big < 128] = (20, 14, 10)
        ok, enc = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            self._image = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()

    def rebaseline(self) -> None:
        self._baseline_redness = None
        self._redness.clear()

    def payload(self) -> dict[str, object]:
        return {
            "image": self._image,
            "box": self._box,
            "ready": self._image is not None,
            "baseline_ready": self._baseline_redness is not None,
            **self._stats,
            "history": [
                [round(a, 1), b, None if c is None else round(c, 2)] for a, b, c in self._history
            ],
        }
