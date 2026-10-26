import { type ReactNode, useEffect, useRef, useState } from "react";

import { clamp01, mean, num, shortDate } from "../lib/format";
import type { Metrics, Tone } from "../lib/types";
import {
  type CheckInSummary,
  currentPeriod,
  type Period,
  PERIODS,
  periodLabel,
  type Profile,
  SKIN_TONE_UNSET,
} from "../lib/useProfile";
import { CloseIcon } from "./Icons";
import { Badge, Empty, Meter, Note, Panel, Stat, Stats } from "./ui";

const DURATION = 60;
const SETTLE = 10;

export function Modal({
  open,
  title,
  onClose,
  children,
}: {
  open: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const dialogRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      previous?.focus();
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div
      className="overlay"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="dialog" role="dialog" aria-modal="true" aria-label={title} tabIndex={-1} ref={dialogRef}>
        <header className="dialog-head">
          <h2>{title}</h2>
          <button type="button" className="btn btn--icon btn--quiet" onClick={onClose} aria-label="Close">
            <CloseIcon />
          </button>
        </header>
        {children}
      </div>
    </div>
  );
}

interface Sample {
  bpm: number | null;
  provisional: number | null;
  rmssd: number | null;
  sdnn: number | null;
  pnn50: number | null;
  respiration: number | null;
  confidence: number;
  usable: boolean;
  lightingMean: number | null;
  lightingDrift: number | null;
  snr: number | null;
  quality: number;
}

function avg(values: (number | null)[]): number | null {
  return mean(values.filter((value): value is number => value !== null));
}

export function CheckInDialog({
  open,
  metrics,
  onClose,
  onComplete,
}: {
  open: boolean;
  metrics: Metrics | null;
  onClose: () => void;
  onComplete: (summary: CheckInSummary) => void;
}) {
  const [phase, setPhase] = useState<"idle" | "running" | "complete">("idle");
  const [elapsed, setElapsed] = useState(0);
  const [summary, setSummary] = useState<CheckInSummary | null>(null);
  const metricsRef = useRef(metrics);
  const samplesRef = useRef<Sample[]>([]);
  const completeRef = useRef(onComplete);
  metricsRef.current = metrics;
  completeRef.current = onComplete;

  useEffect(() => {
    if (open) return;
    setPhase("idle");
    setElapsed(0);
    setSummary(null);
    samplesRef.current = [];
  }, [open]);

  useEffect(() => {
    if (phase !== "running") return;
    const timer = window.setInterval(() => {
      const m = metricsRef.current;
      if (m) {
        samplesRef.current.push({
          bpm: m.bpm,
          provisional: m.provisional_bpm,
          rmssd: m.hrv_ready ? m.rmssd_ms : null,
          sdnn: m.hrv_ready ? m.hrv.sdnn_ms : null,
          pnn50: m.hrv_ready ? m.hrv.pnn50 : null,
          respiration: m.respiration.ready ? m.respiration.bpm : null,
          confidence: m.confidence,
          usable: m.status === "good" || m.status === "measuring",
          lightingMean: m.lighting.mean,
          lightingDrift: m.lighting.drift,
          snr: m.snr_db,
          quality: m.face_quality,
        });
      }
      setElapsed((value) => value + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [phase]);

  useEffect(() => {
    if (phase !== "running" || elapsed < DURATION) return;
    const samples = samplesRef.current.slice(SETTLE);
    const next: CheckInSummary = {
      bpm: avg(samples.map((s) => s.bpm)) ?? avg(samples.map((s) => s.provisional)),
      rmssd_ms: avg(samples.map((s) => s.rmssd)),
      sdnn_ms: avg(samples.map((s) => s.sdnn)),
      pnn50: avg(samples.map((s) => s.pnn50)),
      respiration_bpm: avg(samples.map((s) => s.respiration)),
      confidence: avg(samples.map((s) => s.confidence)) ?? 0,
      coverage: samples.length ? (samples.filter((s) => s.usable).length / samples.length) * 100 : 0,
      duration_s: DURATION,
      created_at: new Date().toISOString(),
      period: currentPeriod(),
      lighting_mean: avg(samples.map((s) => s.lightingMean)),
      lighting_drift: avg(samples.map((s) => s.lightingDrift)),
      snr_db: avg(samples.map((s) => s.snr)),
      quality: avg(samples.map((s) => s.quality)),
    };
    setSummary(next);
    setPhase("complete");
    completeRef.current(next);
  }, [elapsed, phase]);

  const live = metrics;
  const period = periodLabel(currentPeriod()).toLowerCase();

  return (
    <Modal open={open} title="Guided check-in" onClose={onClose}>
      {phase === "idle" ? (
        <div className="dialog-body">
          <p className="lead">Record your {period} resting baseline.</p>
          <ol className="steps">
            <li>Sit still with your face inside the outline.</li>
            <li>Breathe normally. The first 10 seconds are discarded while you settle.</li>
            <li>The remaining 50 seconds are averaged and saved in this browser.</li>
          </ol>
          <div className="dialog-actions">
            <button
              type="button"
              className="btn btn--primary"
              onClick={() => {
                samplesRef.current = [];
                setElapsed(0);
                setPhase("running");
              }}
            >
              Start 60 s check-in
            </button>
          </div>
        </div>
      ) : null}
      {phase === "running" ? (
        <div className="dialog-body">
          <div className="countdown">
            <span className="countdown-value">{Math.max(0, DURATION - elapsed)}</span>
            <span>{elapsed < SETTLE ? "seconds, settling in" : "seconds, breathe normally"}</span>
          </div>
          <Meter value={elapsed / DURATION} tone="info" label="Check-in progress" />
          <Stats columns={3}>
            <Stat label="Heart rate" value={num(live?.bpm ?? live?.provisional_bpm)} unit="BPM" />
            <Stat label="RMSSD" value={live?.hrv_ready ? num(live.rmssd_ms) : "–"} unit="ms" />
            <Stat label="Confidence" value={Math.round((live?.confidence ?? 0) * 100)} unit="%" />
          </Stats>
          <div className="dialog-actions">
            <button type="button" className="btn btn--quiet" onClick={onClose}>
              Cancel check-in
            </button>
          </div>
        </div>
      ) : null}
      {phase === "complete" && summary ? (
        <div className="dialog-body">
          <p className="lead">Baseline saved for the {period}.</p>
          <Stats columns={3}>
            <Stat label="Resting HR" value={num(summary.bpm)} unit="BPM" />
            <Stat label="RMSSD" value={num(summary.rmssd_ms)} unit="ms" />
            <Stat label="SDNN" value={num(summary.sdnn_ms)} unit="ms" />
            <Stat label="pNN50" value={num(summary.pnn50)} unit="%" />
            <Stat label="Respiration" value={num(summary.respiration_bpm)} unit="/min" />
            <Stat label="Usable" value={Math.round(summary.coverage)} unit="%" />
          </Stats>
          {summary.coverage < 60 ? (
            <Note>Less than 60% of the check-in was usable. Consider repeating it in better light.</Note>
          ) : null}
          <div className="dialog-actions">
            <button type="button" className="btn btn--primary" onClick={onClose}>
              Done
            </button>
          </div>
        </div>
      ) : null}
    </Modal>
  );
}

export function BaselinePanel({
  baseline,
  period,
  onStart,
}: {
  baseline: CheckInSummary | null;
  period: Period;
  onStart: () => void;
}) {
  return (
    <Panel
      title={`${periodLabel(period)} baseline`}
      meta={baseline ? `Recorded ${shortDate(baseline.created_at)}, ${Math.round(baseline.coverage)}% usable` : "Not recorded"}
      actions={
        <button type="button" className="btn btn--primary btn--small" onClick={onStart}>
          {baseline ? "Record again" : "Start check-in"}
        </button>
      }
    >
      {baseline ? (
        <Stats columns={4}>
          <Stat label="Resting HR" value={num(baseline.bpm)} unit="BPM" />
          <Stat label="RMSSD" value={num(baseline.rmssd_ms)} unit="ms" />
          <Stat label="SDNN" value={num(baseline.sdnn_ms)} unit="ms" />
          <Stat label="Respiration" value={num(baseline.respiration_bpm)} unit="/min" />
        </Stats>
      ) : (
        <Empty>
          A 60 second check-in records your resting heart rate, HRV and breathing for this time of day.
          The recovery score compares against it. No video is stored.
        </Empty>
      )}
    </Panel>
  );
}

export function RecoveryPanel({
  metrics,
  baseline,
  onCheckIn,
}: {
  metrics: Metrics | null;
  baseline: CheckInSummary | null;
  onCheckIn: () => void;
}) {
  const hrvReady = Boolean(metrics?.hrv_ready && metrics.rmssd_ms !== null);
  const parts: { label: string; value: number; weight: number }[] = [];
  if (baseline?.rmssd_ms && hrvReady && metrics?.rmssd_ms != null) {
    parts.push({ label: "HRV against baseline", value: clamp01(metrics.rmssd_ms / Math.max(1, baseline.rmssd_ms * 1.5)), weight: 0.45 });
    if (baseline.bpm !== null && metrics.bpm !== null) {
      parts.push({ label: "Heart rate against baseline", value: clamp01(1 - Math.abs(metrics.bpm - baseline.bpm) / 20), weight: 0.25 });
    }
    if (baseline.respiration_bpm !== null && metrics.respiration.ready && metrics.respiration.bpm !== null) {
      parts.push({ label: "Breathing against baseline", value: clamp01(1 - Math.abs(metrics.respiration.bpm - baseline.respiration_bpm) / 6), weight: 0.15 });
    }
    parts.push({ label: "Signal confidence", value: clamp01(metrics.confidence), weight: 0.15 });
  }
  const total = parts.reduce((sum, p) => sum + p.weight, 0);
  const score = parts.length ? (parts.reduce((sum, p) => sum + p.value * p.weight, 0) / total) * 100 : null;
  const tone: Tone = score === null ? "idle" : score >= 70 ? "ok" : score >= 45 ? "warn" : "bad";

  return (
    <Panel
      title="Recovery"
      meta={score === null ? undefined : <Badge tone={tone}>{score >= 70 ? "Recovered" : score >= 45 ? "Moderate" : "Low"}</Badge>}
    >
      {!baseline ? (
        <Empty>
          Record a baseline first.{" "}
          <button type="button" className="link" onClick={onCheckIn}>
            Start check-in
          </button>
        </Empty>
      ) : baseline.rmssd_ms === null ? (
        <Empty>
          The saved baseline has no clean HRV.{" "}
          <button type="button" className="link" onClick={onCheckIn}>
            Record it again
          </button>{" "}
          with steadier light.
        </Empty>
      ) : score === null ? (
        <Empty>Waiting for a live HRV reading, which takes about 30 seconds of clean signal.</Empty>
      ) : (
        <>
          <p className="score" data-tone={tone}>
            {Math.round(score)}
            <span>/ 100</span>
          </p>
          <dl className="inputs">
            {parts.map((part) => (
              <div className="input-row" key={part.label}>
                <dt>{part.label}</dt>
                <dd>
                  <Meter value={part.value} tone={tone} label={part.label} />
                  <span className="figure">{Math.round(part.value * 100)}%</span>
                </dd>
              </div>
            ))}
          </dl>
        </>
      )}
      <Note>A wellness indicator relative to your own baseline, not a clinical score.</Note>
    </Panel>
  );
}

export function BreathingPanel({ metrics }: { metrics: Metrics | null }) {
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<"in" | "out">("in");
  const [scale, setScale] = useState(0.6);
  const [cycles, setCycles] = useState(0);

  useEffect(() => {
    if (!running) return;
    const start = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const elapsed = (now - start) / 1000;
      const cycle = elapsed % 10;
      const inhale = cycle < 5;
      const t = inhale ? cycle / 5 : (cycle - 5) / 5;
      const eased = 0.5 - 0.5 * Math.cos(Math.PI * t);
      setPhase(inhale ? "in" : "out");
      setScale(inhale ? 0.6 + 0.4 * eased : 1 - 0.4 * eased);
      setCycles(Math.floor(elapsed / 10));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [running]);

  return (
    <Panel
      title="Paced breathing"
      meta="6 breaths a minute"
      actions={
        <button type="button" className={`btn btn--small ${running ? "btn--quiet" : "btn--primary"}`} onClick={() => setRunning((v) => !v)}>
          {running ? "Stop" : "Start"}
        </button>
      }
    >
      <div className="pacer">
        <div className="pacer-ring" style={{ transform: `scale(${running ? scale : 0.6})` }} />
        <p className="pacer-label">{running ? (phase === "in" ? "Breathe in" : "Breathe out") : "Five seconds in, five out"}</p>
      </div>
      <Stats columns={2}>
        <Stat label="Cycles" value={cycles} />
        <Stat label="Live RMSSD" value={metrics?.hrv_ready ? num(metrics.rmssd_ms) : "–"} unit="ms" />
      </Stats>
    </Panel>
  );
}

const SKIN_TONES = [SKIN_TONE_UNSET, "I", "II", "III", "IV", "V", "VI"];

export function ProfilePanel({
  profile,
  update,
  onReset,
}: {
  profile: Profile;
  update: (patch: Partial<Profile>) => void;
  onReset: () => void;
}) {
  const current = currentPeriod();
  return (
    <Panel title="Profile and baselines" meta="Stored in this browser">
      <div className="form-grid">
        <label className="field">
          <span>Name</span>
          <input value={profile.name} maxLength={40} placeholder="Optional" onChange={(e) => update({ name: e.target.value })} />
        </label>
        <label className="field">
          <span>Skin tone, Fitzpatrick</span>
          <select value={profile.skinTone} onChange={(e) => update({ skinTone: e.target.value })}>
            {SKIN_TONES.map((tone) => (
              <option value={tone} key={tone}>
                {tone === SKIN_TONE_UNSET ? tone : `Type ${tone}`}
              </option>
            ))}
          </select>
        </label>
      </div>
      <label className="toggle">
        <input type="checkbox" checked={profile.showProvisional} onChange={(e) => update({ showProvisional: e.target.checked })} />
        <span>
          Show low-confidence heart rate
          <small>Displays a greyed estimate when the signal is too weak to confirm.</small>
        </span>
      </label>
      <table className="table">
        <thead>
          <tr>
            <th scope="col">Time of day</th>
            <th scope="col">Resting HR</th>
            <th scope="col">RMSSD</th>
            <th scope="col">Recorded</th>
          </tr>
        </thead>
        <tbody>
          {PERIODS.map((item) => {
            const b = profile.baselines[item.value];
            return (
              <tr key={item.value} data-current={item.value === current ? "true" : undefined}>
                <th scope="row">{item.label}</th>
                <td className="figure">{b ? `${num(b.bpm)} BPM` : "–"}</td>
                <td className="figure">{b ? `${num(b.rmssd_ms)} ms` : "–"}</td>
                <td>{b ? shortDate(b.created_at) : "Not yet"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <Note>Skin tone only adjusts lighting guidance. It never changes how the pulse is measured.</Note>
      <div className="panel-footer">
        <button
          type="button"
          className="link link--danger"
          onClick={() => {
            if (window.confirm("Delete the local profile and all baselines?")) onReset();
          }}
        >
          Delete profile and baselines
        </button>
      </div>
    </Panel>
  );
}
