import { useCallback, useEffect, useState } from "react";

import {
  BeatPanel,
  LightingPanel,
  QualityPanel,
  RegionPanel,
  RespirationPanel,
  SpectrumPanel,
  TrendPanel,
} from "./components/Analysis";
import { AthleticsView } from "./components/Athletics";
import { ThermalView } from "./components/Thermal";
import { CopilotPanel, EventLogPanel, HandoverPanel, OverviewPanel } from "./components/Care";
import { Logo } from "./components/Logo";
import { PulseStrip } from "./components/PulseStrip";
import { EyePanel, ResearchPanel, StabilityPanel } from "./components/Research";
import { Segmented } from "./components/ui";
import { VideoStage } from "./components/VideoStage";
import { HeartRateTile, HrvTile, RespirationTile, SignalTile } from "./components/Vitals";
import {
  BaselinePanel,
  BreathingPanel,
  CheckInDialog,
  ProfilePanel,
  RecoveryPanel,
} from "./components/Wellness";
import { api, type AssistantStatus, type CameraInfoItem } from "./lib/api";
import type { Tone } from "./lib/types";
import { useClinura } from "./lib/useClinura";
import { usePoll } from "./lib/usePoll";
import { currentPeriod, useProfile } from "./lib/useProfile";
import { useVitalsHistory } from "./lib/useVitalsHistory";

type View = "monitor" | "athletics" | "thermal" | "handover" | "wellness" | "research";

const VIEWS: { value: View; label: string }[] = [
  { value: "monitor", label: "Monitor" },
  { value: "athletics", label: "Athletics" },
  { value: "thermal", label: "Thermal" },
  { value: "handover", label: "Handover" },
  { value: "wellness", label: "Wellness" },
  { value: "research", label: "Research" },
];

const WINDOWS = [
  { value: 1, label: "1 h" },
  { value: 6, label: "6 h" },
  { value: 12, label: "12 h" },
  { value: 24, label: "24 h" },
];

function initialView(): View {
  const hash = window.location.hash.replace("#", "");
  return VIEWS.some((v) => v.value === hash) ? (hash as View) : "monitor";
}

function engineLabel(connection: string, state: string | undefined, error: string | null | undefined) {
  if (connection !== "live") return { tone: "bad" as Tone, text: connection === "connecting" ? "Connecting" : "Engine offline" };
  if (error || state === "camera_error" || state === "fatal") return { tone: "bad" as Tone, text: "Camera unavailable" };
  if (state === "starting" || !state) return { tone: "warn" as Tone, text: "Starting camera" };
  return { tone: "ok" as Tone, text: "Monitoring" };
}

function CameraPicker({
  active,
  disabled,
  onPick,
}: {
  active: number;
  disabled: boolean;
  onPick: (index: number) => void;
}) {
  const [cameras, setCameras] = useState<CameraInfoItem[]>([]);
  const [help, setHelp] = useState(false);
  const refresh = useCallback(() => {
    api
      .cameras()
      .then((r) => setCameras(r.cameras))
      .catch(() => setCameras([]));
  }, []);
  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 15000);
    return () => window.clearInterval(timer);
  }, [refresh]);
  const list = cameras.some((c) => c.index === active)
    ? cameras
    : [...cameras, { index: active, name: `Camera ${active}` }];
  return (
    <div className="camera-pick">
      <select
        value={active}
        disabled={disabled}
        onChange={(e) => onPick(Number(e.target.value))}
        aria-label="Choose camera"
      >
        {list.map((c) => (
          <option key={c.index} value={c.index}>
            {c.name}
          </option>
        ))}
      </select>
      <button type="button" className="btn btn--quiet btn--small" onClick={refresh} title="Look for cameras again">
        Refresh
      </button>
      <button type="button" className="btn btn--quiet btn--small" onClick={() => setHelp((v) => !v)}>
        iPhone?
      </button>
      {help ? (
        <div className="camera-help" role="note">
          <strong>Use your iPhone as the camera (Continuity Camera)</strong>
          <ol>
            <li>Mac and iPhone signed in to the same Apple ID, Wi-Fi and Bluetooth on.</li>
            <li>On the iPhone: Settings → General → AirPlay &amp; Continuity → turn on Continuity Camera.</li>
            <li>Lock the iPhone and mount it still, rear camera facing you, near the Mac.</li>
            <li>Press Refresh. “iPhone Camera” appears in this list. Pick it.</li>
          </ol>
          <p>Not showing? Plug the iPhone in with a cable, unlock once and tap Trust, then lock it and Refresh.</p>
        </div>
      ) : null}
    </div>
  );
}

function HandoverView({
  hours,
  setHours,
  bed,
  patient,
  update,
}: {
  hours: number;
  setHours: (hours: number) => void;
  bed: string;
  patient: string;
  update: (patch: { bed?: string; patient?: string }) => void;
}) {
  const [assistant, setAssistant] = useState<AssistantStatus | null>(null);
  const summary = usePoll(() => api.telemetry(hours), 15000, [hours]);
  const events = usePoll(() => api.events(hours), 10000, [hours]);

  useEffect(() => {
    api.assistant().then(setAssistant).catch(() => setAssistant(null));
  }, []);

  return (
    <>
      <div className="view-bar">
        <p className="view-intro">
          Telemetry is logged every second while the camera runs. Pick the window the handover and
          questions should cover.
        </p>
        <Segmented label="Reporting window" value={hours} onChange={setHours} options={WINDOWS} />
      </div>
      <div className="columns">
        <div className="column">
          <HandoverPanel
            hours={hours}
            bed={bed}
            patient={patient}
            onBed={(value) => update({ bed: value })}
            onPatient={(value) => update({ patient: value })}
            assistant={assistant}
          />
          <EventLogPanel events={events.data} error={events.error} onAdded={() => void events.refresh()} />
        </div>
        <div className="column column--sticky">
          <OverviewPanel summary={summary.data} hours={hours} />
          <CopilotPanel hours={hours} assistant={assistant} />
        </div>
      </div>
    </>
  );
}

export default function App() {
  const { metrics, hello, frame, connection, mode, setMode, resetSignal, exercise, setCamera } = useClinura();
  const { profile, update, saveBaseline, reset } = useProfile();
  const [view, setView] = useState<View>(initialView);
  const [waveWindow, setWaveWindow] = useState(5);
  const [hours, setHours] = useState(6);
  const [checkInOpen, setCheckInOpen] = useState(false);
  const history = useVitalsHistory(metrics);
  const now = metrics?.timestamp ?? 0;
  const period = currentPeriod();
  const baseline = profile.baselines[period];
  const engine = engineLabel(connection, metrics?.engine_state ?? hello?.state, metrics?.error);

  useEffect(() => {
    const onHash = () => setView(initialView());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (view === "athletics") {
      if (mode !== "body") setMode("body");
    } else if (view !== "research" && mode !== "face") setMode("face");
  }, [view, mode, setMode]);

  const selectView = useCallback(
    (next: View) => {
      setView(next);
      window.history.pushState(null, "", `#${next}`);
    },
    [],
  );

  const lighting = { mean: baseline?.lighting_mean ?? null, drift: baseline?.lighting_drift ?? null };
  const openCheckIn = () => setCheckInOpen(true);

  return (
    <div className="app">
      <a className="skip" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <div className="brand">
          <Logo />
          <span className="brand-name">Clinura</span>
        </div>
        <nav className="tabs" aria-label="Views">
          {VIEWS.map((item) => (
            <button
              key={item.value}
              type="button"
              className="tab"
              aria-current={view === item.value ? "page" : undefined}
              onClick={() => selectView(item.value)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <div className="topbar-status">
          {profile.bed ? <span className="bed">Bed {profile.bed}</span> : null}
          <CameraPicker
            active={metrics?.camera.index ?? 0}
            disabled={connection !== "live"}
            onPick={setCamera}
          />
          <span className="engine" data-tone={engine.tone} title={metrics?.error ?? undefined}>
            <span className="dot" data-tone={engine.tone} />
            {engine.text}
          </span>
        </div>
      </header>

      <main id="main" className="main">
        {view === "monitor" ? (
          <div className="grid">
            <div className="span-8 stage-cell">
              <VideoStage frame={frame} metrics={metrics} connection={connection} mode="face" />
            </div>
            <div className="span-4 vitals">
              <HeartRateTile metrics={metrics} showProvisional={profile.showProvisional} />
              <div className="vitals-pair">
                <HrvTile metrics={metrics} />
                <RespirationTile metrics={metrics} />
              </div>
              <SignalTile metrics={metrics} connection={connection} onReset={resetSignal} />
            </div>
            <div className="span-12">
              <PulseStrip metrics={metrics} windowSec={waveWindow} onWindowChange={setWaveWindow} />
            </div>
            <div className="span-4">
              <TrendPanel points={history} now={now} />
            </div>
            <div className="span-4">
              <BeatPanel metrics={metrics} />
            </div>
            <div className="span-4">
              <QualityPanel points={history} now={now} />
            </div>
            <div className="span-3">
              <SpectrumPanel metrics={metrics} />
            </div>
            <div className="span-3">
              <RegionPanel metrics={metrics} />
            </div>
            <div className="span-3">
              <RespirationPanel metrics={metrics} />
            </div>
            <div className="span-3">
              <LightingPanel metrics={metrics} reference={lighting} skinTone={profile.skinTone} />
            </div>
          </div>
        ) : null}

        {view === "athletics" ? (
          <AthleticsView metrics={metrics} frame={frame} connection={connection} exercise={exercise} />
        ) : null}

        {view === "thermal" ? <ThermalView metrics={metrics} frame={frame} connection={connection} /> : null}

        {view === "handover" ? (
          <HandoverView hours={hours} setHours={setHours} bed={profile.bed} patient={profile.patient} update={update} />
        ) : null}

        {view === "wellness" ? (
          <div className="grid">
            <div className="span-7">
              <BaselinePanel baseline={baseline} period={period} onStart={openCheckIn} />
            </div>
            <div className="span-5 span-rows-2">
              <BreathingPanel metrics={metrics} />
            </div>
            <div className="span-7">
              <RecoveryPanel metrics={metrics} baseline={baseline} onCheckIn={openCheckIn} />
            </div>
            <div className="span-12">
              <ProfilePanel profile={profile} update={update} onReset={reset} />
            </div>
          </div>
        ) : null}

        {view === "research" ? (
          <div className="grid">
            <div className="span-8 stage-cell">
              <VideoStage frame={frame} metrics={metrics} connection={connection} mode={mode} />
            </div>
            <div className="span-4">
              <ResearchPanel metrics={metrics} mode={mode} setMode={setMode} />
            </div>
            <div className="span-4">
              <EyePanel metrics={metrics} />
            </div>
            <div className="span-4">
              <StabilityPanel metrics={metrics} />
            </div>
            <div className="span-4">
              <RegionPanel metrics={metrics} />
            </div>
          </div>
        ) : null}
      </main>

      <footer className="footer">
        <span>Clinura {hello?.version ?? ""}</span>
        <span>
          Research prototype. Readings are camera-derived estimates, not a medical device. Video stays on
          this computer.
        </span>
      </footer>

      <CheckInDialog
        open={checkInOpen}
        metrics={metrics}
        onClose={() => setCheckInOpen(false)}
        onComplete={saveBaseline}
      />
    </div>
  );
}
