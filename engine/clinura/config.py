from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
POSE_MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"
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
class PoseConfig:
    model_path: Path = POSE_MODEL_PATH
    detection_width: int = 960
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class SignalConfig:
    # Windows
    hr_window_s: float = 10.0
    hrv_window_s: float = 60.0
    min_hr_window_s: float = 8.0
    min_hrv_window_s: float = 30.0
    max_gap_s: float = 0.5
    analysis_interval_s: float = 0.0667
    min_sample_quality: float = 0.08
    reset_after_loss_s: float = 2.0

    # Heart-rate search range (Hz). 0.65-3.5 Hz = 39-210 BPM, wide enough for
    # athletes at rest and for hard exercise.
    band_low_hz: float = 0.65
    band_high_hz: float = 3.5
    # Pulse band-pass; wider than the search range so the first harmonic survives
    # (needed for the de Haan SNR and harmonic-aware tracking).
    filter_low_hz: float = 0.55
    filter_high_hz: float = 4.0
    filter_order: int = 3
    spectrum_high_hz: float = 4.0
    welch_nfft: int = 4096
    # Kept for estimate_hr callers (experimental pipeline, scripts).
    welch_nperseg: int = 256

    # Rgb -> pulse
    pos_window_s: float = 1.6
    artifact_clip_sigma: float = 3.5
    fuse_rois: bool = True
    use_chrom: bool = True
    channel_weight_tau_s: float = 3.0

    # Tracking and display smoothing
    tracker_drift_bpm: float = 3.0
    tracker_jump_cost: float = 2.5
    tracker_obs_tau_s: float = 1.0
    # Harmonic bonus favours half-rate errors (58 BPM for a 117 BPM heart) with real
    # webcam noise, so it is off by default.
    harmonic_weight: float = 0.0
    whitening: float = 0.5
    whitening_width_hz: float = 0.8
    # SNR is measured on a noise-flattened spectrum so slow rates (where webcam noise
    # is strongest) are not favoured over fast ones by the lock gate.
    snr_whitening: float = 1.0
    slow_rate_bpm: float = 50.0
    refine_span_bpm: float = 6.0
    smoother_process_bpm: float = 1.5
    smoother_base_noise_bpm: float = 2.0

    # Motion suppression from head movement
    motion_threshold: float = 0.004
    motion_full_scale: float = 0.012
    motion_max_suppression: float = 0.8

    # Status gating (with hysteresis so the reading does not flicker).
    # A reading is trusted when the SNR is clearly good, or when it is moderate but
    # several independent skin regions/methods agree on the rate and the tracker has
    # held the same peak for a few seconds.
    min_snr_db: float = -1.0
    good_snr_db: float = 3.0
    assisted_snr_db: float = -4.5
    min_agreement: float = 0.25
    assisted_stable_s: float = 3.0
    agreement_tolerance_bpm: float = 5.0
    provisional_min_snr_db: float = -8.0
    confidence_snr_low_db: float = -5.0
    snr_tau_s: float = 1.5
    lock_hysteresis_db: float = 1.5
    lock_confirm_s: float = 1.0
    hold_s: float = 4.0

    # Beats and HRV
    peak_prominence_std: float = 0.25
    min_ibi_s: float = 0.28
    max_ibi_s: float = 1.6
    max_ibi_deviation: float = 0.20
    min_valid_ibis: int = 8
    peak_upsample: int = 4
    beat_template: bool = True
    beat_min_correlation: float = 0.6
    hrv_bias_correction: bool = True
    hrv_timing_noise_factor: float = 0.6
    hrv_smoothing: int = 5
    hrv_interval_s: float = 1.0
    hrv_rate_tolerance_bpm: float = 8.0

    # Respiration
    respiration_window_s: float = 32.0
    min_respiration_window_s: float = 16.0
    respiration_interval_s: float = 1.0
    respiration_low_hz: float = 0.1
    respiration_high_hz: float = 0.75
    respiration_min_quality: float = 0.22
    respiration_min_snr_db: float = -3.0


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
    athletics_path: Path | None = DATA_DIR / "athletics.db"
    # Required when listening on a non-loopback address (see security.py).
    access_token: str | None = None


@dataclass(frozen=True)
class RecordConfig:
    enabled: bool = False
    output_dir: Path = SESSION_DIR
    video_fourcc: str = "FFV1"
    video_ext: str = ".avi"
