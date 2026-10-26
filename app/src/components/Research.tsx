import { num } from "../lib/format";
import type { Metrics, MonitorMode, Tone } from "../lib/types";
import { Chart } from "./Chart";
import { Badge, Meter, Note, Panel, Segmented, Stat, Stats } from "./ui";

export const MODE_OPTIONS: { value: MonitorMode; label: string }[] = [
  { value: "face", label: "Face" },
  { value: "tongue", label: "Tongue" },
  { value: "palm", label: "Palm" },
];

export function ResearchPanel({
  metrics,
  mode,
  setMode,
}: {
  metrics: Metrics | null;
  mode: MonitorMode;
  setMode: (mode: MonitorMode) => void;
}) {
  const exp = metrics?.experimental ?? null;
  const bpm = exp?.bpm ?? exp?.provisional_bpm ?? null;
  const provisional = exp?.bpm === null && exp?.provisional_bpm !== null;
  return (
    <Panel
      title="Measurement site"
      actions={<Segmented label="Measurement site" value={mode} onChange={setMode} options={MODE_OPTIONS} />}
    >
      {mode === "face" ? (
        <dl className="site-list">
          <div>
            <dt>Face</dt>
            <dd>Forehead and both cheeks. The primary measurement path.</dd>
          </div>
          <div>
            <dt>Tongue</dt>
            <dd>Inner mouth with colour segmentation. Experimental.</dd>
          </div>
          <div>
            <dt>Palm</dt>
            <dd>Hand landmarks define the palm. Experimental.</dd>
          </div>
        </dl>
      ) : (
        <>
          <div className="research-readout">
            <p className="vital-value vital-value--small" data-provisional={provisional ? "true" : undefined}>
              {bpm === null ? "– –" : Math.round(bpm)}
              <span className="vital-unit"> BPM</span>
            </p>
            <Badge tone={exp?.active ? "ok" : "warn"}>{exp?.active ? "Region locked" : "Searching"}</Badge>
          </div>
          <p className="muted">{exp?.message ?? "Waiting for the region"}</p>
          <Stats columns={3}>
            <Stat label="SNR" value={num(exp?.snr_db, 1)} unit="dB" />
            <Stat label="Quality" value={exp?.quality == null ? "–" : Math.round(exp.quality * 100)} unit="%" />
            <Stat label={exp?.handedness ? "Hand" : "Pixels"} value={exp?.handedness ?? (exp?.pixels ? exp.pixels.toLocaleString() : "–")} />
          </Stats>
          <Chart
            label="Experimental pulse waveform"
            x={exp?.waveform.t ?? []}
            y={exp?.waveform.y ?? []}
            xRange={[-10, 0]}
            color={mode === "tongue" ? "--roi-tongue" : "--roi-palm"}
            height={110}
            xTick={(t) => `${t.toFixed(0)} s`}
            empty="Waveform appears once the region is locked"
          />
          <Note>Unvalidated. The vitals on the Monitor view always come from the face.</Note>
        </>
      )}
    </Panel>
  );
}

export function EyePanel({ metrics }: { metrics: Metrics | null }) {
  const eye = metrics?.eye;
  return (
    <Panel title="Eye activity">
      <Stats columns={3}>
        <Stat label="Blink rate" value={num(eye?.blink_rate)} unit="/min" />
        <Stat label="Blinks, 60 s" value={eye?.blinks ?? 0} />
        <Stat label="Openness" value={eye?.openness == null ? "–" : Math.round(eye.openness * 100)} unit="%" />
      </Stats>
      <Meter value={eye?.openness ?? 0} tone="info" label="Eye openness" />
      <Note>From face blendshapes. Long closures are logged as eyes-closed rest periods.</Note>
    </Panel>
  );
}

export function StabilityPanel({ metrics }: { metrics: Metrics | null }) {
  const s = metrics?.stability;
  const score = s?.score ?? 0;
  const tone: Tone = !metrics?.face_detected ? "idle" : score >= 70 ? "ok" : score >= 45 ? "warn" : "bad";
  return (
    <Panel title="Head stability" meta={metrics?.face_detected ? <Badge tone={tone}>{score >= 70 ? "Still" : score >= 45 ? "Some movement" : "Moving"}</Badge> : undefined}>
      <Stats columns={3}>
        <Stat label="Score" value={metrics?.face_detected ? Math.round(score) : "–"} unit="/ 100" tone={tone} />
        <Stat label="Motion" value={num(s?.motion_px_s)} unit="px/s" />
        <Stat label="Yaw, roll" value={s ? `${Math.round(s.yaw_ratio * 100)}%, ${Math.round(s.roll_deg)}°` : "–"} />
      </Stats>
      <Meter value={score / 100} tone={tone} label="Head stability" />
    </Panel>
  );
}
