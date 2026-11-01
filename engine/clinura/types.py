from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ROI:
    name: str
    contour: np.ndarray
    mean_rgb: tuple[float, float, float]
    pixel_count: int


@dataclass
class FaceObservation:
    timestamp: float
    detected: bool
    landmarks: np.ndarray | None = None
    bbox: tuple[int, int, int, int] | None = None
    rois: dict[str, ROI] = field(default_factory=dict)
    face_contour: np.ndarray | None = None
    quality: float = 0.0
    pose_ok: bool = True
    reason: str | None = None
    frame_size: tuple[int, int] = (0, 0)
    face_width_px: float = 0.0
    yaw_ratio: float = 0.0
    roll_deg: float = 0.0
    mean_luma: float = 0.0
    blink_left: float = 0.0
    blink_right: float = 0.0
    head_xy: tuple[float, float] | None = None
    head_pose: tuple[float, float, float] | None = None

    @property
    def combined_rgb(self) -> tuple[float, float, float] | None:
        rois = [r for r in self.rois.values() if r.pixel_count > 0]
        if not rois:
            return None
        weights = np.asarray([r.pixel_count for r in rois], dtype=np.float64)
        rgbs = np.asarray([r.mean_rgb for r in rois], dtype=np.float64)
        return tuple(float(v) for v in np.average(rgbs, axis=0, weights=weights))

    @property
    def roi_pixels(self) -> int:
        return int(sum(r.pixel_count for r in self.rois.values()))


@dataclass
class SignalSample:
    timestamp: float
    rgb: tuple[float, float, float]
    quality: float
    face_width_px: float
    roi_pixels: int
    roi_rgb: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    roi_pixels_by_name: dict[str, int] = field(default_factory=dict)
    luma: float = 0.0
    head_xy: tuple[float, float] | None = None


@dataclass
class AnalysisResult:
    timestamp: float
    status: str
    message: str
    wall_time: float | None = None
    bpm: float | None = None
    provisional_bpm: float | None = None
    snr_db: float | None = None
    rmssd_ms: float | None = None
    hrv_ready: bool = False
    hrv_progress: float = 0.0
    confidence: float = 0.0
    fs: float | None = None
    window_s: float = 0.0
    sample_count: int = 0
    ibi_count: int = 0
    waveform_t: np.ndarray | None = None
    waveform_y: np.ndarray | None = None
    psd_f: np.ndarray | None = None
    psd_p: np.ndarray | None = None
    raw_t: np.ndarray | None = None
    raw_rgb: np.ndarray | None = None
    beats: np.ndarray | None = None
    analysis_ms: float | None = None
    hrv_sdnn_ms: float | None = None
    hrv_pnn50: float | None = None
    hrv_mean_ibi_ms: float | None = None
    ibi_times: np.ndarray | None = None
    ibis_ms: np.ndarray | None = None
    roi_quality: list[dict[str, object]] = field(default_factory=list)
    luma_mean: float | None = None
    luma_std: float | None = None
    respiration_bpm: float | None = None
    respiration_snr_db: float | None = None
    respiration_ready: bool = False
    respiration_progress: float = 0.0
    respiration_t: np.ndarray | None = None
    respiration_y: np.ndarray | None = None
    respiration_sources: list[dict[str, object]] = field(default_factory=list)
    motion_level: float | None = None
    lock_state: str = "searching"
    hrv_quality: float | None = None
