import { clamp01, mean, num, snrTone } from "../lib/format";
import type { Metrics, Tone } from "../lib/types";
import type { HistoryPoint } from "../lib/useVitalsHistory";
import { Chart } from "./Chart";
import { Badge, Empty, Meter, Note, Panel, Stat, Stats } from "./ui";

interface MetricsProps {
  metrics: Metrics | null;
}

const ROI_LABELS: Record<string, string> = {
  forehead: "Forehead",
  right_cheek: "Right cheek",
  left_cheek: "Left cheek",
};

const ROI_TOKENS: Record<string, string> = {
  forehead: "var(--roi-forehead)",
  right_cheek: "var(--roi-right)",
  left_cheek: "var(--roi-left)",
};

export function SpectrumPanel({ metrics }: MetricsProps) {
  const freqs = metrics?.psd.f ?? [];
  const db = (metrics?.psd.p ?? []).map((value) => 10 * Math.log10(Math.max(value, 1e-12)));
  const peakBpm = metrics?.bpm ?? metrics?.provisional_bpm ?? null;
  return (
    <Panel
      title="Spectrum"
      meta={peakBpm === null ? "Welch PSD" : `Peak ${(peakBpm / 60).toFixed(2)} Hz`}
    >
      <Chart
        label="Pulse power spectrum"
        x={freqs}
        y={db}
        xRange={[0.5, 3.5]}
        color="--hr"
        fill
        height={150}
        showY
        xTick={(v) => `${v.toFixed(1)} Hz`}
        yTick={(v) => `${v.toFixed(0)}`}
        vLines={peakBpm === null ? [] : [peakBpm / 60]}
        empty="Builds after 8 s of signal"
      />
      <Note>Power in dB. The marked line is the reported heart rate.</Note>
    </Panel>
  );
}

export function RegionPanel({ metrics }: MetricsProps) {
  const rois = metrics?.roi_quality ?? [];
  const best = rois.reduce<(typeof rois)[number] | null>(
    (top, roi) => (roi.snr_db !== null && (top === null || roi.snr_db > (top.snr_db ?? -Infinity)) ? roi : top),
    null,
  );
  return (
    <Panel title="Skin regions" meta={best ? `Strongest: ${ROI_LABELS[best.name] ?? best.name}` : undefined}>
      {rois.length ? (
        <ul className="regions">
          {rois.map((roi) => {
            const tone = snrTone(roi.snr_db);
            return (
              <li key={roi.name}>
                <span className="swatch" style={{ background: ROI_TOKENS[roi.name] ?? "var(--ink-3)" }} />
                <span className="region-name">{ROI_LABELS[roi.name] ?? roi.name}</span>
                <span className="figure region-bpm">{roi.bpm === null ? "–" : `${roi.bpm.toFixed(0)} BPM`}</span>
                <span className="figure region-snr" data-tone={tone}>
                  {roi.snr_db === null ? "–" : `${roi.snr_db.toFixed(1)} dB`}
                </span>
                <Meter value={roi.snr_db === null ? 0 : clamp01((roi.snr_db + 6) / 12)} tone={tone} label={`${roi.name} SNR`} />
              </li>
            );
          })}
        </ul>
      ) : (
        <Empty>Appears once all three regions are tracked</Empty>
      )}
      <Note>Regions are fused by signal-to-noise ratio, so a shadowed cheek contributes little.</Note>
    </Panel>
  );
}

export function BeatPanel({ metrics }: MetricsProps) {
  const ready = metrics?.hrv_ready ?? false;
  const times = metrics?.hrv.ibi_times ?? [];
  const ibis = metrics?.hrv.ibis_ms ?? [];
  const count = Math.min(times.length, ibis.length);
  const avg = mean(ibis.slice(0, count));
  const v = (value: number | null | undefined) => (ready ? num(value) : "–");
  return (
    <Panel title="Beat to beat" meta={ready ? `${count} intervals, 60 s` : "Needs 30 s of clean signal"}>
      <Stats columns={4}>
        <Stat label="RMSSD" value={v(metrics?.rmssd_ms)} unit="ms" />
        <Stat label="SDNN" value={v(metrics?.hrv.sdnn_ms)} unit="ms" />
        <Stat label="pNN50" value={v(metrics?.hrv.pnn50)} unit="%" />
        <Stat label="Mean IBI" value={v(metrics?.hrv.mean_ibi_ms)} unit="ms" />
      </Stats>
      <Chart
        label="Inter-beat interval tachogram"
        x={times.slice(0, count)}
        y={ibis.slice(0, count)}
        xRange={[-60, 0]}
        color="--hrv"
        points
        height={112}
        showY
        xTick={(t) => `${t.toFixed(0)} s`}
        yTick={(t) => t.toFixed(0)}
        hLines={avg === null ? [] : [avg]}
        empty="Intervals appear when HRV is ready"
      />
    </Panel>
  );
}

export function RespirationPanel({ metrics }: MetricsProps) {
  const respiration = metrics?.respiration;
  return (
    <Panel
      title="Breathing"
      meta={respiration?.snr_db == null ? "45 s window" : `SNR ${respiration.snr_db.toFixed(1)} dB`}
    >
      <Chart
        label="Respiration waveform"
        x={respiration?.t ?? []}
        y={respiration?.y ?? []}
        xRange={[-45, 0]}
        color="--resp"
        fill
        height={150}
        xTick={(t) => `${t.toFixed(0)} s`}
        empty="Builds after 20 s of signal"
      />
      <Note>Derived from slow changes in skin colour. Beta, not a respiratory measurement.</Note>
    </Panel>
  );
}

export function TrendPanel({ points, now }: { points: HistoryPoint[]; now: number }) {
  const x = points.map((point) => point.t - now);
  const y = points.map((point) => (point.provisional ? null : point.bpm));
  const confirmed = y.filter((value): value is number => value !== null);
  const avg = mean(confirmed);
  return (
    <Panel title="Heart rate, last minute" meta="Confirmed readings only">
      <Stats columns={4}>
        <Stat label="Now" value={confirmed.length ? Math.round(confirmed[confirmed.length - 1]) : "–"} />
        <Stat label="Low" value={confirmed.length ? Math.round(Math.min(...confirmed)) : "–"} />
        <Stat label="High" value={confirmed.length ? Math.round(Math.max(...confirmed)) : "–"} />
        <Stat label="Average" value={avg === null ? "–" : Math.round(avg)} />
      </Stats>
      <Chart
        label="Heart rate over the last minute"
        x={x}
        y={y}
        xRange={[-60, 0]}
        color="--hr"
        height={112}
        showY
        xTick={(t) => `${t.toFixed(0)} s`}
        yTick={(t) => t.toFixed(0)}
        hLines={avg === null ? [] : [avg]}
        empty="No confirmed readings yet"
      />
    </Panel>
  );
}

const STATUS_LABELS: Record<string, string> = {
  good: "Good",
  measuring: "Measuring",
  low_signal: "Weak signal",
  adjust_pose: "Turned away",
  no_face: "No face",
  error: "Error",
  acquiring: "Collecting",
};

export function QualityPanel({ points, now }: { points: HistoryPoint[]; now: number }) {
  const slots = Array.from({ length: 60 }, (_, index) => {
    const target = now - (59 - index);
    let best: HistoryPoint | null = null;
    let distance = Infinity;
    for (const point of points) {
      const d = Math.abs(point.t - target);
      if (d < distance) {
        distance = d;
        best = point;
      }
    }
    return distance <= 1.2 ? best : null;
  });
  const available = slots.filter((slot): slot is HistoryPoint => slot !== null);
  const usable = available.filter((s) => s.status === "good" || s.status === "measuring").length;
  const coverage = available.length ? Math.round((usable / available.length) * 100) : null;
  const tone: Tone = coverage === null ? "idle" : coverage >= 80 ? "ok" : coverage >= 50 ? "warn" : "bad";

  return (
    <Panel title="Signal quality, last minute" meta={coverage === null ? undefined : <Badge tone={tone}>{coverage}% usable</Badge>}>
      <div className="quality-strip" role="img" aria-label={`Signal usable ${coverage ?? 0}% of the last minute`}>
        {slots.map((slot, index) => (
          <span
            key={index}
            className="quality-cell"
            data-status={slot?.status ?? "none"}
            title={slot ? `${STATUS_LABELS[slot.status] ?? slot.status}, ${Math.max(1, Math.round(now - slot.t))} s ago` : "No data"}
          />
        ))}
      </div>
      <div className="legend">
        <span><i data-status="good" />Good</span>
        <span><i data-status="measuring" />Measuring</span>
        <span><i data-status="low_signal" />Weak</span>
        <span><i data-status="no_face" />No face</span>
      </div>
      <Chart
        label="Signal-to-noise ratio over the last minute"
        x={points.map((p) => p.t - now)}
        y={points.map((p) => p.snr)}
        xRange={[-60, 0]}
        color="--ink-2"
        height={86}
        showY
        xTick={(t) => `${t.toFixed(0)} s`}
        yTick={(t) => `${t.toFixed(0)}`}
        hLines={[0, 3]}
        empty="Collecting"
      />
      <Note>SNR in dB. Readings are reported above 0 dB and marked good above 3 dB.</Note>
    </Panel>
  );
}

export function LightingPanel({
  metrics,
  reference,
  skinTone,
}: MetricsProps & { reference: { mean: number | null; drift: number | null }; skinTone: string }) {
  const luma = metrics?.lighting.mean ?? null;
  const drift = metrics?.lighting.drift ?? null;
  const darker = skinTone === "V" || skinTone === "VI";
  const low = darker ? 90 : 70;
  const lumaTone: Tone = luma === null ? "idle" : luma < 55 || luma > 215 ? "bad" : luma < low || luma > 200 ? "warn" : "ok";
  const driftTone: Tone = drift === null ? "idle" : drift > 3 ? "bad" : drift > 1.5 ? "warn" : "ok";
  const verdict =
    luma === null
      ? "Waiting"
      : lumaTone === "ok" && driftTone === "ok"
        ? "Steady"
        : lumaTone === "bad"
          ? luma < 55
            ? "Too dark"
            : "Too bright"
          : driftTone === "bad"
            ? "Exposure drifting"
            : "Adjust lighting";
  const worst: Tone = [lumaTone, driftTone].includes("bad") ? "bad" : [lumaTone, driftTone].includes("warn") ? "warn" : lumaTone;
  return (
    <Panel title="Lighting" meta={<Badge tone={worst}>{verdict}</Badge>}>
      <Stats columns={2}>
        <Stat label="Face brightness" value={num(luma)} unit="/ 255" tone={lumaTone} />
        <Stat label="Exposure drift" value={num(drift, 1)} tone={driftTone} />
      </Stats>
      <div className="range">
        <Meter value={luma === null ? 0 : luma / 255} tone={lumaTone} label="Face brightness" />
        <span
          className="range-target"
          style={{ left: `${(low / 255) * 100}%`, width: `${((200 - low) / 255) * 100}%` }}
          aria-hidden="true"
        />
      </div>
      <Note>
        Target {low}–200{darker ? ", raised for darker skin" : ""}. Drift above 1.5 means the camera
        is changing exposure.
        {reference.mean !== null ? ` Your baseline: ${Math.round(reference.mean)}.` : ""}
      </Note>
    </Panel>
  );
}
