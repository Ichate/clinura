import { clamp01, num, scoreTone } from "../lib/format";
import type { ConnectionState, Metrics, Tone } from "../lib/types";
import { CheckIcon } from "./Icons";
import { Badge, Meter } from "./ui";

interface VitalProps {
  metrics: Metrics | null;
}

function heartState(metrics: Metrics | null, shown: boolean, provisional: boolean) {
  if (provisional) return "Low confidence, shown for reference";
  if (shown) return (metrics?.confidence ?? 0) > 0.72 ? "Stable reading" : "Settling";
  switch (metrics?.status) {
    case "low_signal":
      return "Signal too weak to report";
    case "adjust_pose":
      return "Face the camera to continue";
    case "no_face":
      return "No face in view";
    case "error":
      return "Camera unavailable";
    default:
      return metrics?.window_s
        ? `Collecting ${num(metrics.window_s)} of 8 s`
        : "Collecting signal";
  }
}

export function HeartRateTile({
  metrics,
  showProvisional,
}: VitalProps & { showProvisional: boolean }) {
  const bpm = metrics?.bpm ?? null;
  const provisionalBpm = showProvisional ? (metrics?.provisional_bpm ?? null) : null;
  const shown = bpm ?? provisionalBpm;
  const provisional = bpm === null && provisionalBpm !== null;
  const confidence = clamp01(metrics?.confidence ?? 0);

  return (
    <section className="vital vital--hr" aria-live="polite" aria-label="Heart rate">
      <div className="vital-head">
        <h2 className="vital-name">Heart rate</h2>
        <span className="vital-unit">BPM</span>
      </div>
      <p className="vital-value" data-provisional={provisional ? "true" : undefined}>
        {shown === null ? "– –" : Math.round(shown)}
      </p>
      <p className="vital-state">{heartState(metrics, shown !== null, provisional)}</p>
      <div className="vital-foot">
        <span>Confidence</span>
        <Meter
          value={confidence}
          tone={confidence >= 0.55 ? "ok" : "warn"}
          label="Measurement confidence"
        />
        <span className="figure">{Math.round(confidence * 100)}%</span>
      </div>
    </section>
  );
}

export function HrvTile({ metrics }: VitalProps) {
  const ready = metrics?.hrv_ready ?? false;
  const progress = clamp01(metrics?.hrv_progress ?? 0);
  return (
    <section className="vital vital--hrv" aria-label="Heart-rate variability">
      <div className="vital-head">
        <h2 className="vital-name">HRV, RMSSD</h2>
        <span className="vital-unit">ms</span>
      </div>
      <p className="vital-value vital-value--small">
        {ready && metrics?.rmssd_ms != null ? Math.round(metrics.rmssd_ms) : "– –"}
      </p>
      {ready ? (
        <p className="vital-state">{metrics?.ibi_count ?? 0} clean intervals</p>
      ) : (
        <>
          <p className="vital-state">
            {progress >= 0.999 ? "Waiting for clean beats" : `Needs 30 s, ${Math.round(progress * 100)}%`}
          </p>
          <Meter value={progress} tone="info" label="HRV window progress" />
        </>
      )}
    </section>
  );
}

export function RespirationTile({ metrics }: VitalProps) {
  const respiration = metrics?.respiration;
  const ready = respiration?.ready ?? false;
  const progress = clamp01(respiration?.progress ?? 0);
  return (
    <section className="vital vital--resp" aria-label="Respiration rate">
      <div className="vital-head">
        <h2 className="vital-name">Respiration</h2>
        <span className="vital-unit">/min</span>
      </div>
      <p className="vital-value vital-value--small" data-provisional={!ready && respiration?.bpm != null ? "true" : undefined}>
        {respiration?.bpm != null ? Math.round(respiration.bpm) : "– –"}
      </p>
      {ready ? (
        <p className="vital-state">Beta estimate</p>
      ) : (
        <>
          <p className="vital-state">
            {progress >= 0.999 ? "Weak breathing signal" : `Needs 20 s, ${Math.round(progress * 100)}%`}
          </p>
          <Meter value={progress} tone="info" label="Respiration window progress" />
        </>
      )}
    </section>
  );
}

function coaching(metrics: Metrics | null, connected: boolean): string[] {
  if (!connected) return ["Start the engine with python main.py", "This page reconnects on its own"];
  if (!metrics) return ["Waiting for the first measurement"];
  if (metrics.error) {
    return [
      "Close other apps that may be using the camera",
      "Allow camera access for your terminal in System Settings",
    ];
  }
  if (metrics.face_detected && !metrics.pose_ok) {
    return ["Face the camera directly", "Keep your head level"];
  }
  switch (metrics.status) {
    case "no_face":
      return ["Centre your face in the outline", "Use soft light from the front"];
    case "low_signal":
      return ["Add light to your face", "Stay still and avoid talking", "Move a little closer"];
    case "acquiring":
      return ["Stay still while the window fills", "Avoid touching your face"];
    case "good":
      return ["Signal locked", "Keep still for a clean HRV reading"];
    default:
      return ["Hold still while the reading settles", "Avoid strong light behind you"];
  }
}

export function SignalTile({
  metrics,
  connection,
  onReset,
}: VitalProps & { connection: ConnectionState; onReset: () => void }) {
  const connected = connection === "live";
  const snr = metrics?.snr_db ?? null;
  const inputs = [
    { key: "snr", label: "Pulse SNR", value: snr === null ? "–" : `${snr.toFixed(1)} dB`, score: snr === null ? 0 : clamp01(snr / 3) },
    { key: "face", label: "Face quality", value: `${Math.round((metrics?.face_quality ?? 0) * 100)}%`, score: clamp01(metrics?.face_quality ?? 0) },
    { key: "window", label: "Window", value: `${num(metrics?.window_s ?? 0)} of 10 s`, score: clamp01((metrics?.window_s ?? 0) / 10) },
  ];
  const limiting = inputs.reduce((low, item) => (item.score < low.score ? item : low), inputs[0]);
  const live = connected && metrics?.face_detected && metrics.pose_ok && !metrics.error;
  const advice =
    limiting.score >= 0.66
      ? "All inputs are strong."
      : limiting.key === "snr"
        ? "The pulse is weak against camera noise. Add front light or reduce movement."
        : limiting.key === "face"
          ? "The face is small or turned. Move closer and face the camera."
          : "Collecting a full analysis window.";
  const locked = metrics?.status === "good";
  const tone: Tone = !connected || metrics?.error ? "bad" : locked ? "ok" : metrics?.face_detected ? "warn" : "idle";

  return (
    <section className="vital vital--signal" aria-label="Signal quality">
      <div className="vital-head">
        <h2 className="vital-name">Signal</h2>
        <Badge tone={tone}>
          {!connected ? "Offline" : metrics?.error ? "Camera error" : locked ? "Locked" : metrics?.face_detected ? "Tracking" : "Searching"}
        </Badge>
        <button type="button" className="btn btn--quiet btn--small" onClick={onReset} disabled={!connected}>
          Restart reading
        </button>
      </div>
      {live ? (
        <>
          <dl className="inputs">
            {inputs.map((item) => (
              <div className="input-row" key={item.key}>
                <dt>{item.label}</dt>
                <dd>
                  <Meter value={item.score} tone={scoreTone(item.score)} label={item.label} />
                  <span className="figure">{item.value}</span>
                </dd>
              </div>
            ))}
          </dl>
          <p className="limiting" data-tone={scoreTone(limiting.score)}>
            {limiting.score >= 0.66 ? advice : `${limiting.label} is limiting. ${advice}`}
          </p>
        </>
      ) : null}
      <ul className="coach">
        {coaching(metrics, connected).map((tip) => (
          <li key={tip}>
            {locked ? <CheckIcon className="coach-icon" /> : <span className="coach-mark" aria-hidden="true" />}
            {tip}
          </li>
        ))}
      </ul>
    </section>
  );
}
