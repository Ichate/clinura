"""Thermal mode: radiometric thermal cameras.

A normal webcam cannot measure temperature. This mode reads a *radiometric* thermal
camera (every pixel is a temperature) running alongside the RGB camera:

* ``lepton`` – FLIR Lepton 3.x on a PureThermal board (UVC Y16, centi-kelvin);
* ``tc001``  – TOPDON TC001 / InfiRay P2 Pro style (YUYV frame whose lower half holds
  16-bit raw values, kelvin x 64);
* ``simulated`` – a synthetic face for trying the mode without hardware.

Features (all computed on the temperature map, not on colours):

* face = largest region warmer than ``face_min_c``;
* **inner canthus** (eye corner) = hottest point in the upper half of the face. Fever
  screening standards (ISO/TR 13154, IEC 80601-2-59) use it because it tracks core
  temperature best; the **core estimate** is canthus + a configurable offset, plus the
  user's calibration against a reference thermometer;
* forehead mean, maximum face temperature, ambient (background median);
* **fever alert** with hysteresis;
* **breathing rate** from nostril/mouth temperature: exhaled air warms the skin below the
  nose and inhaled air cools it. The face's lower-middle grid cell with the strongest
  periodic signal is used, then the same spectral tracker as the rPPG respiration.
"""

from __future__ import annotations

import base64
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .dsp import SpectralTracker, refine_near
from .dsp.respiration import make_source

logger = logging.getLogger(__name__)

FORMATS = ("auto", "lepton", "tc001", "simulated")


@dataclass
class ThermalSettings:
    enabled: bool = False
    camera_index: int = 1
    format: str = "auto"
    fever_c: float = 37.5
    canthus_offset_c: float = 0.3
    calibration_c: float = 0.0
    face_min_c: float = 30.0


@dataclass
class ThermalReading:
    timestamp: float
    temps: np.ndarray
    face_box: tuple[int, int, int, int] | None = None
    canthus: tuple[int, int] | None = None
    nose_box: tuple[int, int, int, int] | None = None
    max_c: float | None = None
    canthus_c: float | None = None
    forehead_c: float | None = None
    ambient_c: float | None = None
    core_c: float | None = None
    breathing_signal: float | None = None
    extra: dict[str, object] = field(default_factory=dict)


# ----------------------------------------------------------------- sources


class ThermalSource:
    """Background reader returning temperature frames in °C."""

    def __init__(self, settings: ThermalSettings) -> None:
        self.settings = settings
        self.name = settings.format
        self._cap: cv2.VideoCapture | None = None
        self._latest: tuple[float, np.ndarray] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self._sim_t0 = time.perf_counter()

    def start(self) -> None:
        if self.settings.format != "simulated":
            cap = cv2.VideoCapture(self.settings.camera_index)
            if not cap.isOpened():
                raise RuntimeError(
                    f"Thermal camera {self.settings.camera_index} could not be opened."
                )
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            fmt = self.settings.format
            if fmt == "lepton":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
            self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="clinura-thermal", daemon=True)
        self._thread.start()

    def _decode(self, frame: np.ndarray) -> np.ndarray:
        fmt = self.settings.format
        if frame.dtype == np.uint16 and fmt in ("auto", "lepton"):
            self.name = "lepton"
            return frame.astype(np.float32) / 100.0 - 273.15
        raw = np.asarray(frame)
        if fmt in ("auto", "tc001"):
            # YUYV delivered unconverted: (H, W, 2) or flattened bytes. Lower half = data.
            if raw.ndim == 2 and raw.shape[0] == 1:
                raw = raw.reshape(-1)
            if raw.ndim == 1 and raw.size % (256 * 2) == 0:
                raw = raw.reshape(-1, 256, 2)
            if raw.ndim == 3 and raw.shape[2] == 2 and raw.shape[0] >= 2:
                half = raw.shape[0] // 2
                data = raw[half:].astype(np.uint16)
                values = data[:, :, 0] + (data[:, :, 1] << 8)
                self.name = "tc001"
                return values.astype(np.float32) / 64.0 - 273.15
        raise RuntimeError(
            "This camera is not sending radiometric data. Pick the thermal camera's index and "
            "format (Lepton/PureThermal or TC001/P2). A normal webcam cannot measure temperature."
        )

    def _simulate(self) -> np.ndarray:
        t = time.perf_counter() - self._sim_t0
        h, w = 120, 160
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        temps = (
            23.0
            + 0.3 * np.sin(xx / 25.0)
            + np.random.default_rng(int(t * 9)).normal(0, 0.05, (h, w))
        )
        cx, cy = 80 + 3 * math.sin(t / 7), 62
        face = ((xx - cx) / 30) ** 2 + ((yy - cy) / 40) ** 2 <= 1
        temps[face] = 34.2 + 0.6 * (1 - ((xx[face] - cx) / 30) ** 2)
        for ex in (cx - 11, cx + 11):  # inner canthi, warmest
            eye = ((xx - ex) ** 2 + (yy - (cy - 10)) ** 2) <= 9
            temps[eye] = 36.7 + 0.05 * math.sin(t / 30)
        breath = math.sin(2 * math.pi * (15 / 60) * t)
        nose = ((xx - cx) ** 2 / 36 + (yy - (cy + 12)) ** 2 / 9) <= 1
        temps[nose] = 33.4 + 0.45 * breath
        return temps.astype(np.float32)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.settings.format == "simulated":
                temps = self._simulate()
                self._stop.wait(1 / 9)
            else:
                assert self._cap is not None
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self.error = "Thermal camera stopped delivering frames."
                    self._stop.wait(0.2)
                    continue
                try:
                    temps = self._decode(frame)
                    self.error = None
                except RuntimeError as exc:
                    self.error = str(exc)
                    self._stop.wait(0.5)
                    continue
            with self._lock:
                self._latest = (time.perf_counter(), temps)

    def latest(self) -> tuple[float, np.ndarray] | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------- analysis


def analyze_frame(temps: np.ndarray, settings: ThermalSettings, t: float) -> ThermalReading:
    reading = ThermalReading(timestamp=t, temps=temps)
    finite = np.isfinite(temps)
    if not finite.any():
        return reading
    warm = ((temps >= settings.face_min_c) & (temps <= 43.0)).astype(np.uint8)
    warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(warm, connectivity=8)
    background = temps[(warm == 0) & finite]
    reading.ambient_c = float(np.median(background)) if background.size else None
    if count <= 1:
        return reading
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < max(30, 0.005 * temps.size):
        return reading
    x, y, w, h = (int(v) for v in stats[largest, :4])
    reading.face_box = (x, y, w, h)
    face_mask = labels == largest
    face_temps = temps[face_mask]
    reading.max_c = float(np.percentile(face_temps, 99.5))

    upper = face_mask.copy()
    upper[y + h // 2 :, :] = False
    if upper.any():
        masked = np.where(upper, temps, -np.inf)
        cy, cx = np.unravel_index(int(np.argmax(masked)), temps.shape)
        reading.canthus = (int(cx), int(cy))
        y0, y1 = max(0, cy - 1), min(temps.shape[0], cy + 2)
        x0, x1 = max(0, cx - 1), min(temps.shape[1], cx + 2)
        reading.canthus_c = float(np.max(temps[y0:y1, x0:x1]))
        reading.core_c = reading.canthus_c + settings.canthus_offset_c + settings.calibration_c

    forehead = face_mask.copy()
    forehead[: y + int(0.05 * h), :] = False
    forehead[y + int(0.28 * h) :, :] = False
    forehead[:, : x + int(0.25 * w)] = False
    forehead[:, x + int(0.75 * w) :] = False
    if forehead.sum() >= 4:
        reading.forehead_c = float(np.median(temps[forehead]) + settings.calibration_c)
    return reading


@dataclass
class _NoseGrid:
    box: tuple[int, int, int, int]
    history: deque = field(default_factory=lambda: deque(maxlen=600))


class ThermalMonitor:
    """Owns the thermal source and turns frames into readings, breathing and alerts."""

    def __init__(self, on_event=None) -> None:
        self.settings = ThermalSettings()
        self._source: ThermalSource | None = None
        self._lock = threading.Lock()
        self._last_frame_t = 0.0
        self._reading: ThermalReading | None = None
        self._trend: deque[tuple[float, float | None, float | None]] = deque(maxlen=600)
        self._last_trend = 0.0
        self._cells: list[deque[tuple[float, float]]] = [deque(maxlen=400) for _ in range(9)]
        self._face_size: tuple[int, int] | None = None
        self._tracker = SpectralTracker(
            4.0, 45.0, step=0.25, drift_per_sqrt_s=1.2, jump_cost=3.0, obs_tau_s=2.0
        )
        self._breathing: dict[str, object] = {"bpm": None, "ready": False, "t": [], "y": []}
        self._last_breath = 0.0
        self._fever = False
        self._fever_count = 0
        self._image_cache: tuple[float, str] | None = None
        self.error: str | None = None
        self._on_event = on_event

    # -------------------------------------------------------------- control

    def configure(self, **changes: object) -> None:
        restart = False
        with self._lock:
            for key, value in changes.items():
                if value is None or not hasattr(self.settings, key):
                    continue
                if (
                    key in ("enabled", "camera_index", "format")
                    and getattr(self.settings, key) != value
                ):
                    restart = True
                setattr(self.settings, key, value)
        if self.settings.format not in FORMATS:
            self.settings.format = "auto"
        if restart:
            self._restart()

    def calibrate(self, reference_c: float) -> float | None:
        reading = self._reading
        if reading is None or reading.canthus_c is None:
            return None
        self.settings.calibration_c = round(
            reference_c - (reading.canthus_c + self.settings.canthus_offset_c), 2
        )
        return self.settings.calibration_c

    def _restart(self) -> None:
        if self._source is not None:
            self._source.stop()
            self._source = None
        self._reset_state()
        self.error = None
        if not self.settings.enabled:
            return
        source = ThermalSource(self.settings)
        try:
            source.start()
        except RuntimeError as exc:
            self.error = str(exc)
            return
        self._source = source

    def _reset_state(self) -> None:
        self._reading = None
        for cell in self._cells:
            cell.clear()
        self._tracker.reset()
        self._breathing = {"bpm": None, "ready": False, "t": [], "y": []}
        self._fever = False
        self._fever_count = 0

    def stop(self) -> None:
        if self._source is not None:
            self._source.stop()
            self._source = None

    # -------------------------------------------------------------- update

    def poll(self) -> None:
        source = self._source
        if source is None:
            return
        latest = source.latest()
        self.error = source.error
        if latest is None or latest[0] == self._last_frame_t:
            return
        t, temps = latest
        self._last_frame_t = t
        reading = analyze_frame(temps, self.settings, t)
        self._reading = reading
        self._update_breathing(reading, temps, t)
        self._update_fever(reading, t)
        if t - self._last_trend >= 1.0:
            self._last_trend = t
            self._trend.append((time.time(), reading.core_c, reading.max_c))

    def _update_breathing(self, reading: ThermalReading, temps: np.ndarray, t: float) -> None:
        if reading.face_box is None:
            for cell in self._cells:
                cell.clear()
            return
        x, y, w, h = reading.face_box
        # 3x3 grid over the lower-middle face (nose and mouth).
        x0, x1 = x + int(0.25 * w), x + int(0.75 * w)
        y0, y1 = y + int(0.50 * h), y + int(0.95 * h)
        cw, ch = max(1, (x1 - x0) // 3), max(1, (y1 - y0) // 3)
        for i in range(3):
            for j in range(3):
                patch = temps[y0 + i * ch : y0 + (i + 1) * ch, x0 + j * cw : x0 + (j + 1) * cw]
                if patch.size:
                    self._cells[3 * i + j].append((t, float(np.mean(patch))))
        if t - self._last_breath < 1.0:
            return
        self._last_breath = t
        best = None
        for index, cell in enumerate(self._cells):
            if len(cell) < 60:
                continue
            ts = np.asarray([c[0] for c in cell])
            ys = np.asarray([c[1] for c in cell])
            if ts[-1] - ts[0] < 15:
                continue
            fs = (len(ts) - 1) / (ts[-1] - ts[0])
            grid = np.arange(ts[0], ts[-1], 1 / fs)
            source = make_source("nostril", grid, np.interp(grid, ts, ys), fs, 0.1, 0.75)
            if source is not None and (best is None or source.quality > best[1].quality):
                best = (index, source)
        if best is None:
            return
        index, source = best
        state = self._tracker.update(source.freqs * 60.0, source.power, t)
        rate = refine_near(source.freqs * 60.0, source.power, state.rate, 2.5)
        i, j = divmod(index, 3)
        reading.nose_box = (x0 + j * cw, y0 + i * ch, cw, ch)
        self._breathing = {
            "bpm": round(rate, 1),
            "ready": bool(source.quality >= 0.3 and state.stable_s >= 3.0),
            "quality": round(source.quality, 2),
            "t": [round(float(v - t), 2) for v in source.t[::2]],
            "y": [round(float(v), 3) for v in source.y[::2]],
        }

    def _update_fever(self, reading: ThermalReading, t: float) -> None:
        core = reading.core_c
        if core is None:
            return
        threshold = self.settings.fever_c
        if not self._fever:
            self._fever_count = self._fever_count + 1 if core >= threshold else 0
            if self._fever_count >= 20:  # ~2 s of consistent readings
                self._fever = True
                if self._on_event:
                    self._on_event("fever", f"Elevated temperature {core:.1f} °C", core)
        elif core < threshold - 0.3:
            self._fever = False
            self._fever_count = 0
            if self._on_event:
                self._on_event("fever_end", f"Temperature back below {threshold:.1f} °C", core)

    # ------------------------------------------------------------- payload

    def _image(self, reading: ThermalReading) -> str:
        now = reading.timestamp
        if self._image_cache is not None and now - self._image_cache[0] < 0.12:
            return self._image_cache[1]
        temps = reading.temps
        lo = float(np.nanpercentile(temps, 2))
        hi = float(np.nanpercentile(temps, 99.8))
        norm = np.clip((temps - lo) / max(hi - lo, 0.5) * 255, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        scale = max(1, 480 // max(1, temps.shape[1]))
        color = cv2.resize(color, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ok, encoded = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = (
            "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode() if ok else ""
        )
        self._image_cache = (now, data)
        return data

    def payload(self) -> dict[str, object]:
        s = self.settings
        reading = self._reading
        base: dict[str, object] = {
            "enabled": s.enabled,
            "settings": {
                "camera_index": s.camera_index,
                "format": s.format,
                "fever_c": s.fever_c,
                "canthus_offset_c": s.canthus_offset_c,
                "calibration_c": s.calibration_c,
            },
            "source": None if self._source is None else self._source.name,
            "error": self.error,
            "live": reading is not None and time.perf_counter() - reading.timestamp < 2.0,
        }
        if reading is None:
            return base

        def r(v: float | None) -> float | None:
            return None if v is None or not math.isfinite(v) else round(v, 2)

        h, w = reading.temps.shape
        base.update(
            {
                "width": w,
                "height": h,
                "image": self._image(reading),
                "face_box": reading.face_box,
                "canthus": reading.canthus,
                "nose_box": reading.nose_box,
                "max_c": r(reading.max_c),
                "canthus_c": r(reading.canthus_c),
                "core_c": r(reading.core_c),
                "forehead_c": r(reading.forehead_c),
                "ambient_c": r(reading.ambient_c),
                "fever": self._fever,
                "breathing": self._breathing,
                "trend": [
                    [round(t, 1), r(core), r(mx)] for t, core, mx in list(self._trend)[-600:]
                ],
            }
        )
        return base
