from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DATA_DIR = PROJECT_ROOT / "data"
SESSION_DIR = DATA_DIR / "sessions"


@dataclass(frozen=True)
class CaptureConfig:
    camera_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    mirror: bool = True
    warmup_s: float = 0.8
    manual_exposure: bool = False
    exposure_value: float = -6.0
    stall_timeout_s: float = 3.0
    black_level: float = 4.0
    black_frame_timeout_s: float = 4.0
    auto_select: bool = True
    max_camera_index: int = 3


@dataclass(frozen=True)
class FaceConfig:
    model_path: Path = MODEL_PATH
    min_face_detection_confidence: float = 0.35
    min_face_presence_confidence: float = 0.35
    min_tracking_confidence: float = 0.45
    num_faces: int = 1
    smoothing: float = 0.55
    detection_width: int = 1280
    min_face_width_px: int = 56
    max_yaw_ratio: float = 0.30
    max_roll_deg: float = 25.0
    fallback_crop: bool = True
    fallback_crop_width_ratio: float = 0.62
    fallback_crop_height_ratio: float = 0.82
    fallback_input_width: int = 1280


@dataclass(frozen=True)
class HandConfig:
    model_path: Path = HAND_MODEL_PATH
    detection_width: int = 1280
    min_hand_detection_confidence: float = 0.45
    min_hand_presence_confidence: float = 0.45
    min_tracking_confidence: float = 0.45
    num_hands: int = 1


@dataclass(frozen=True)
class SignalConfig:
    hr_window_s: float = 10.0
    hrv_window_s: float = 60.0
    min_hr_window_s: float = 8.0
    min_hrv_window_s: float = 30.0
    band_low_hz: float = 0.75
    band_high_hz: float = 2.8
    filter_order: int = 3
    welch_nperseg: int = 256
    welch_nfft: int = 4096
    peak_prominence_std: float = 0.25
    min_ibi_s: float = 0.33
    max_ibi_s: float = 1.5
    max_ibi_deviation: float = 0.20
    min_valid_ibis: int = 8
    max_gap_s: float = 0.5
    analysis_interval_s: float = 0.0667
    min_snr_db: float = 0.0
    good_snr_db: float = 3.0
    provisional_min_snr_db: float = -8.0
    bpm_smoothing: int = 9
    hrv_smoothing: int = 3
    hrv_interval_s: float = 1.0
    respiration_window_s: float = 45.0
    min_respiration_window_s: float = 20.0
    respiration_interval_s: float = 1.0
    respiration_low_hz: float = 0.1
    respiration_high_hz: float = 0.5
    respiration_min_snr_db: float = 0.0
    min_sample_quality: float = 0.08
    pos_window_s: float = 1.6
    artifact_clip_sigma: float = 3.5
    fuse_rois: bool = True
    fusion_min_snr_db: float = -6.0
    prior_tolerance_hz: float = 0.13
    prior_min_relative_power: float = 0.5
    max_prior_overrides: int = 30
    hrv_rate_tolerance_bpm: float = 8.0
    reset_after_loss_s: float = 2.0


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 3000
    preview_width: int = 900
    preview_fps: float = 30.0
    preview_jpeg_quality: int = 80
    metrics_hz: float = 15.0
    camera_retry_s: float = 3.0
    telemetry_path: Path | None = DATA_DIR / "telemetry.db"


@dataclass(frozen=True)
class RecordConfig:
    enabled: bool = False
    output_dir: Path = SESSION_DIR
    video_fourcc: str = "FFV1"
    video_ext: str = ".avi"
