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

export type MonitorMode = "face" | "tongue" | "palm" | "body";

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
  body: BodyMetrics | null;
  perfusion?: {
    image: string | null;
    ready: boolean;
    mode?: "none" | "redness" | "perfusion";
    progress?: number;
    baseline_ready: boolean;
    perfusion_index?: number;
    asymmetry_pct?: number;
    warmer_side?: string | null;
    redness?: number;
    flush_delta?: number | null;
    coverage_pct?: number;
    history: [number, number | null, number | null][];
  } | null;
  thermal?: {
    enabled: boolean;
    live: boolean;
    error: string | null;
    image?: string;
    core_c?: number | null;
    max_c?: number | null;
    fever?: boolean;
    breathing?: { bpm: number | null; ready: boolean };
  } | null;
  lock_state?: "searching" | "locked" | "holding";
  motion_level?: number | null;
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

export interface RepLog {
  index: number;
  side: string;
  duration_s: number;
  rom_min: number;
  rom_max: number;
  rom: number;
  form_score: number;
  issues: string[];
}

export interface ExerciseLive {
  exercise: string;
  kind: "reps" | "hold";
  side: string;
  active_side: string | null;
  reps: number;
  partials: number;
  hold_s: number;
  hold_best_s: number;
  angle: number | null;
  phase: string | null;
  visible: boolean;
  cues: string[];
  avg_rom: number | null;
  best_rom: number | null;
  avg_form: number | null;
  avg_rep_s: number | null;
  last_rep: RepLog | null;
  rep_log: RepLog[];
}

export interface BodyMetrics {
  available: boolean;
  error: string | null;
  detected: boolean;
  landmarks: [number, number, number][] | null;
  skeleton: [number, number][];
  hand?: [number, number][] | null;
  session: ExerciseLive | null;
}

export interface ExerciseSpec {
  key: string;
  name: string;
  category: string;
  kind: "reps" | "hold";
  source?: "pose" | "hand";
  seated?: boolean;
  joint: string;
  start_angle: number;
  target_angle: number;
  per_side: boolean;
  default_sets: number;
  default_reps: number;
  default_hold_s: number;
  default_rest_s: number;
  setup: string;
  cues: string[];
}

export interface PlanItem {
  id: number;
  program_id: number;
  exercise: string;
  exercise_name: string;
  kind: "reps" | "hold";
  sets: number;
  reps: number;
  hold_s: number;
  rest_s: number;
  side: string;
  days: string;
  times: string[];
  target_rom: number | null;
  max_hr: number | null;
  instructions: string | null;
  position: number;
}

export interface Program {
  id: number;
  title: string;
  patient: string | null;
  prescriber: string | null;
  prescriber_role: string | null;
  goal: string | null;
  notes: string | null;
  start_date: string | null;
  end_date: string | null;
  active: boolean;
  items: PlanItem[];
}

export interface ScheduleSlot {
  date: string;
  time: string | null;
  status: "done" | "partial" | "due" | "upcoming" | "overdue" | "missed";
  session_id: number | null;
  program: { id: number; title: string; prescriber: string | null; prescriber_role: string | null };
  item: PlanItem;
}

export interface AdherenceDay {
  date: string;
  scheduled: number;
  done: number;
  partial: number;
}

export interface ExerciseSession {
  id?: number;
  program_id: number | null;
  item_id: number | null;
  exercise: string;
  exercise_name?: string;
  slot_date: string | null;
  slot_time: string | null;
  started_at: number;
  ended_at: number;
  sets_done: number;
  reps_done: number;
  reps_target: number;
  hold_s: number | null;
  partials: number;
  avg_form: number | null;
  avg_rom: number | null;
  best_rom: number | null;
  hr_pre: number | null;
  hr_peak: number | null;
  hr_post: number | null;
  hr_recovery: number | null;
  rpe: number | null;
  pain: number | null;
  notes: string | null;
  completed: boolean;
  detail?: unknown;
}
