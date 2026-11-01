import { useEffect, useState } from "react";

import { num } from "../lib/format";
import type { ConnectionState, Metrics } from "../lib/types";
import { Chart } from "./Chart";
import { Badge, Empty, Note, Panel, Stat, Stats } from "./ui";
import { VideoStage } from "./VideoStage";

async function post(path: string, body: object) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const data = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new Error(data.detail ?? `Request failed (${response.status})`);
  }
  return response.json();
}

function flushLabel(delta: number | null | undefined) {
  if (delta == null) return "Setting baseline…";
  if (delta > 1.5) return "Flushed vs start";
  if (delta < -1.5) return "Paler vs start";
  return "Same as start";
}

export function ThermalView({
  metrics,
  frame,
  connection,
}: {
  metrics: Metrics | null;
  frame: ImageBitmap | null;
  connection: ConnectionState;
}) {
  const [error, setError] = useState<string | null>(null);
  const [showDevice, setShowDevice] = useState(false);
  const [device, setDevice] = useState({ camera_index: 1, format: "auto" });
  const [reference, setReference] = useState("");
  const perf = metrics?.perfusion ?? null;
  const thermal = metrics?.thermal ?? null;

  useEffect(() => {
    post("/api/perfusion", { enabled: true }).catch(() => undefined);
    return () => {
      post("/api/perfusion", { enabled: false }).catch(() => undefined);
    };
  }, []);

  const act = (work: Promise<unknown>) => {
    setError(null);
    work.catch((exc: unknown) => setError(exc instanceof Error ? exc.message : "Request failed."));
  };

  const history = perf?.history ?? [];
  const t0 = history.length ? history[history.length - 1][0] : 0;

  const mode = perf?.mode ?? "none";
  const tone = mode === "perfusion" ? "ok" : mode === "redness" ? "info" : "idle";
  const asym = perf?.asymmetry_pct ?? null;

  return (
    <>
      <div className="view-bar">
        <p className="view-intro">
          Skin blood-flow map from your webcam, shown on a thermal colour scale. It is not a temperature reading.
        </p>
      </div>
      <div className="grid">
        <div className="span-8">
          <section className="thermal-hero" aria-label="Facial heat map">
            {perf?.image ? (
              <img src={perf.image} alt="Facial blood-flow heat map" />
            ) : (
              <div className="thermal-empty">
                <p className="stage-notice-title">Face the camera</p>
                <p className="stage-notice-copy">The map appears as soon as your face is found.</p>
              </div>
            )}
            <div className="thermal-hero-top">
              <Badge tone={tone}>{mode === "perfusion" ? "Blood-flow map" : mode === "redness" ? "Quick view: redness" : "Searching"}</Badge>
              {mode !== "perfusion" && perf ? (
                <span className="thermal-progress">
                  <span style={{ width: `${Math.round((perf.progress ?? 0) * 100)}%` }} />
                </span>
              ) : null}
            </div>
            <div className="thermal-legend">
              <span>Low</span>
              <span className="thermal-bar" />
              <span>High</span>
            </div>
          </section>
        </div>
        <div className="span-4 thermal-side">
          <div className="thermal-preview">
            <VideoStage frame={frame} metrics={metrics} connection={connection} mode="face" />
          </div>
          <div className="thermal-cards">
            <div className="thermal-card">
              <span className="thermal-card-label">Blood flow</span>
              <span className="thermal-card-value figure">{perf?.ready ? num(perf.perfusion_index) : "–"}<small>%</small></span>
            </div>
            <div className="thermal-card" data-tone={asym != null && asym >= 25 ? "warn" : undefined}>
              <span className="thermal-card-label">Left / right</span>
              <span className="thermal-card-value figure">{perf?.ready ? num(asym) : "–"}<small>%</small></span>
              <span className="thermal-card-note">{perf?.warmer_side ? `More on ${perf.warmer_side}` : "Balanced"}</span>
            </div>
            <div className="thermal-card" data-tone={(perf?.flush_delta ?? 0) > 1.5 ? "warn" : undefined}>
              <span className="thermal-card-label">Redness</span>
              <span className="thermal-card-value figure">
                {perf?.flush_delta == null ? "–" : `${perf.flush_delta > 0 ? "+" : ""}${num(perf.flush_delta, 1)}`}
              </span>
              <span className="thermal-card-note">{flushLabel(perf?.flush_delta)}</span>
            </div>
          </div>
          <button type="button" className="btn btn--small" onClick={() => act(post("/api/perfusion", { rebaseline: true }))}>
            Reset redness baseline
          </button>
        </div>
        <div className="span-6">
          <Panel title="Trend" meta="Last 10 minutes">
            <Chart
              label="Perfusion index trend"
              x={history.map((h) => h[0] - t0)}
              y={history.map((h) => h[1])}
              xRange={[-600, 0]}
              color="--hr"
              height={120}
              showY
              xTick={(v) => `${Math.round(v / 60)} min`}
              empty="Builds while this tab is open"
            />
            <Chart
              label="Redness change trend"
              x={history.map((h) => h[0] - t0)}
              y={history.map((h) => h[2])}
              xRange={[-600, 0]}
              color="--resp"
              height={100}
              showY
              hLines={[0]}
              xTick={(v) => `${Math.round(v / 60)} min`}
              empty="Redness change appears after the baseline"
            />
          </Panel>
        </div>
        <div className="span-6">
          <Panel title="How to read it">
            <ul className="cue-list">
              <li>Bright areas pulse strongly: well-perfused, warmer skin. Cheeks and forehead are usually brightest.</li>
              <li>A large left/right difference can mean one-sided flushing or uneven light — check the light first.</li>
              <li>Redness rises after exercise, heat or stress and falls with cold or pallor.</li>
              <li>This is not a temperature. Use a thermometer for fever.</li>
            </ul>
            {error ? <Note>{error}</Note> : null}
            <button type="button" className="link" onClick={() => setShowDevice((v) => !v)}>
              {showDevice ? "Hide" : "Have a real thermal camera?"}
            </button>
            {showDevice ? (
              <div className="stack thermal-device">
                <Note>Radiometric USB thermal cameras (FLIR Lepton / PureThermal, TOPDON TC001, InfiRay P2) give real temperatures.</Note>
                <div className="form-row">
                  <label className="field">
                    <span>Camera index</span>
                    <input type="number" min={0} max={15} value={device.camera_index} onChange={(e) => setDevice({ ...device, camera_index: Number(e.target.value) })} />
                  </label>
                  <label className="field">
                    <span>Format</span>
                    <select value={device.format} onChange={(e) => setDevice({ ...device, format: e.target.value })}>
                      <option value="auto">Auto</option>
                      <option value="lepton">Lepton / PureThermal</option>
                      <option value="tc001">TC001 / P2</option>
                      <option value="simulated">Simulated (demo)</option>
                    </select>
                  </label>
                  <button type="button" className="btn" onClick={() => act(post("/api/thermal", { enabled: !thermal?.enabled, ...device }))}>
                    {thermal?.enabled ? "Disconnect" : "Connect"}
                  </button>
                </div>
                {thermal?.error ? <Note>{thermal.error}</Note> : null}
                {thermal?.enabled && thermal.live ? (
                  <>
                    {thermal.image ? <img className="thermal-raw" src={thermal.image} alt="Thermal camera image" /> : null}
                    <Stats columns={3}>
                      <Stat label="Core estimate" value={num(thermal.core_c, 1)} unit="°C" tone={thermal.fever ? "bad" : "ok"} />
                      <Stat label="Max face" value={num(thermal.max_c, 1)} unit="°C" />
                      <Stat label="Breathing" value={num(thermal.breathing?.bpm)} unit="/min" />
                    </Stats>
                    {thermal.fever ? <Note>Elevated temperature — confirm with a thermometer.</Note> : null}
                    <form
                      className="form-row"
                      onSubmit={(e) => {
                        e.preventDefault();
                        act(post("/api/thermal/calibrate", { reference_c: Number(reference) }));
                      }}
                    >
                      <label className="field">
                        <span>Thermometer reading (°C)</span>
                        <input type="number" step={0.1} min={34} max={42} value={reference} onChange={(e) => setReference(e.target.value)} />
                      </label>
                      <button type="submit" className="btn btn--small">
                        Calibrate
                      </button>
                    </form>
                  </>
                ) : thermal?.enabled ? (
                  <Empty>Waiting for thermal frames…</Empty>
                ) : null}
              </div>
            ) : null}
          </Panel>
        </div>
      </div>
    </>
  );
}
