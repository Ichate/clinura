export type SignalStatus =
  | "no_face"
  | "acquiring"
  | "low_signal"
  | "adjust_pose"
  | "measuring"
  | "good"
  | "error";

export type DisplayStatus = SignalStatus | "offline";

export type EngineState = "starting" | "running" | "camera_error" | "fatal" | "stopped";

export type MonitorMode = "face" | "tongue" | "palm";

export type Tone = "ok" | "warn" | "bad" | "idle" | "info";

export interface Series {
  t: number[];
  y: number[];
}

export interface Psd {
  f: number[];
  p: number[];
}

export interface Roi {
  name: string;
  pixels: number;
  contour: [number, number][];
}

export interface ExperimentalMetrics {
  mode: MonitorMode;
  active: boolean;
  status: string;
  message: string;
  bpm: number | null;
  provisional_bpm: number | null;
  snr_db: number | null;
  confidence: number;
  quality: number | null;
  handedness: string | null;
  pixels: number;
  roi: Roi | null;
  waveform: Series;
  psd: Psd;
  window_s: number | null;
  analysis_ms: number | null;
}

export interface RoiQuality {
  name: string;
  bpm: number | null;
  snr_db: number | null;
  pixels: number;
}

export interface HrvMetrics {
  sdnn_ms: number | null;
  pnn50: number | null;
  mean_ibi_ms: number | null;
  ibi_times: number[];
  ibis_ms: number[];
}

export interface RespirationMetrics extends Series {
  bpm: number | null;
  snr_db: number | null;
  ready: boolean;
  progress: number;
}

export interface CameraInfo {
  width: number;
  height: number;
  fps: number;
  index?: number;
}

export interface Metrics {
  type: "metrics";
  timestamp: number;
  engine_state: EngineState;
  status: SignalStatus;
  message: string;
  error: string | null;
  bpm: number | null;
  provisional_bpm: number | null;
  snr_db: number | null;
  rmssd_ms: number | null;
  hrv_ready: boolean;
  hrv_progress: number;
  confidence: number;
  fs: number | null;
  window_s: number;
  sample_count: number;
  ibi_count: number;
  analysis_ms: number | null;
  face_detected: boolean;
  pose_ok: boolean;
  face_reason: string | null;
  face_bbox: [number, number, number, number] | null;
  face_contour: [number, number][] | null;
  face_width_px: number;
  roi_pixels: number;
  face_quality: number;
  yaw_ratio: number | null;
  roll_deg: number | null;
  rois: Roi[];
  psd: Psd;
  roi_quality: RoiQuality[];
  lighting: { mean: number | null; drift: number | null };
  hrv: HrvMetrics;
  respiration: RespirationMetrics;
  mode: MonitorMode;
  experimental: ExperimentalMetrics | null;
  eye: { blink_rate: number | null; blinks: number; openness: number | null };
  stability: {
    score: number;
    motion_px_s: number | null;
    yaw_ratio: number;
    roll_deg: number;
  };
  camera: CameraInfo;
  waveform: Series;
  raw: { t: number[]; r: number[]; g: number[]; b: number[] };
  beats: number[];
}

export interface Hello {
  type: "hello";
  version: string;
  state: EngineState;
  camera: CameraInfo;
  model: string;
}

export type ConnectionState = "connecting" | "live" | "reconnecting" | "offline";

export interface ClinicalEvent {
  id: number;
  ts: number;
  time: string;
  clock: string;
  end_ts: number | null;
  end_clock: string | null;
  duration_s: number | null;
  ongoing: boolean;
  kind: string;
  severity: "info" | "warning" | "critical";
  title: string;
  detail: string | null;
  value: number | null;
  source: "detector" | "manual";
}

export interface StatBlock {
  mean: number;
  sd: number;
  min: number;
  max: number;
  median: number;
}

export interface TelemetryBin {
  from: string;
  hr_mean: number | null;
  hr_max: number | null;
  rmssd_mean: number | null;
  resp_mean: number | null;
  stability_mean: number | null;
  eyes_closed_pct: number | null;
  hr_coverage_pct: number;
}

export interface TelemetrySummary {
  window: {
    start: string;
    end: string;
    hours: number;
    first_sample: string | null;
    seconds_recorded: number;
  };
  coverage: { patient_in_view_pct: number; confirmed_hr_pct: number };
  heart_rate_bpm: StatBlock | null;
  heart_rate_peak: { bpm: number; time: string } | null;
  heart_rate_lowest: { bpm: number; time: string } | null;
  rmssd_ms: StatBlock | null;
  respiration_bpm: StatBlock | null;
  series_bin_seconds: number;
  series: TelemetryBin[];
}

export interface HandoverResult {
  text: string;
  generated_at: string;
  generated_by: string;
  window: TelemetrySummary["window"];
  coverage: TelemetrySummary["coverage"];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  error?: boolean;
}
