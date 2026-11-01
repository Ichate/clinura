import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { athletics, setClinicianPin } from "../lib/api";
import { num } from "../lib/format";
import type {
  AdherenceDay,
  ConnectionState,
  ExerciseLive,
  ExerciseSession,
  ExerciseSpec,
  Metrics,
  PlanItem,
  Program,
  RepLog,
  ScheduleSlot,
  Tone,
} from "../lib/types";
import { usePoll } from "../lib/usePoll";
import { Badge, Empty, Meter, Note, Panel, Segmented, Stat, Stats } from "./ui";
import { VideoStage } from "./VideoStage";

type Role = "patient" | "clinician";
type ExerciseCommand = (action: "start" | "stop", key?: string, side?: string) => void;

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const ROLES = ["Physician", "Physiotherapist", "Trainer", "Coach", "Nurse"];
const STATUS_TONE: Record<ScheduleSlot["status"], Tone> = {
  done: "ok",
  partial: "warn",
  due: "info",
  upcoming: "idle",
  overdue: "bad",
  missed: "bad",
};

interface Run {
  spec: ExerciseSpec;
  sets: number;
  reps: number;
  hold_s: number;
  rest_s: number;
  side: string;
  max_hr: number | null;
  target_rom: number | null;
  instructions: string | null;
  program_id: number | null;
  item_id: number | null;
  slot_date: string | null;
  slot_time: string | null;
}

function runFromItem(spec: ExerciseSpec, item: PlanItem | null, slot: ScheduleSlot | null): Run {
  return {
    spec,
    sets: item?.sets ?? spec.default_sets,
    reps: item?.reps ?? spec.default_reps,
    hold_s: item?.hold_s || spec.default_hold_s,
    rest_s: item?.rest_s ?? spec.default_rest_s,
    side: item?.side ?? "auto",
    max_hr: item?.max_hr ?? null,
    target_rom: item?.target_rom ?? null,
    instructions: item?.instructions ?? null,
    program_id: item?.program_id ?? null,
    item_id: item?.id ?? null,
    slot_date: slot?.date ?? null,
    slot_time: slot?.time ?? null,
  };
}

function dose(item: { kind: string; sets: number; reps: number; hold_s: number }) {
  return item.kind === "hold" ? `${item.sets} × ${item.hold_s} s hold` : `${item.sets} × ${item.reps} reps`;
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function speak(text: string) {
  try {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.1;
    window.speechSynthesis.speak(utterance);
  } catch {
    return;
  }
}

/* ------------------------------------------------------------------ view */

export function AthleticsView({
  metrics,
  frame,
  connection,
  exercise,
}: {
  metrics: Metrics | null;
  frame: ImageBitmap | null;
  connection: ConnectionState;
  exercise: ExerciseCommand;
}) {
  const [role, setRole] = useState<Role>("patient");
  const [library, setLibrary] = useState<ExerciseSpec[]>([]);
  const [run, setRun] = useState<Run | null>(null);
  const today = usePoll(() => athletics.today(), 30000);
  const adherence = usePoll(() => athletics.adherence(14), 60000);
  const sessions = usePoll(() => athletics.sessions(30), 60000);

  useEffect(() => {
    athletics.library().then(setLibrary).catch(() => setLibrary([]));
  }, []);

  const refreshAll = useCallback(() => {
    void today.refresh();
    void adherence.refresh();
    void sessions.refresh();
  }, [today, adherence, sessions]);

  const specFor = useCallback((key: string) => library.find((e) => e.key === key), [library]);

  return (
    <>
      <div className="view-bar">
        <p className="view-intro">
          Exercises prescribed by a physician, physiotherapist or trainer. The camera tracks your body to count
          reps, measure range of motion and check form; heart rate is read from your face between sets.
        </p>
        <Segmented
          label="Who is using Clinura"
          value={role}
          onChange={setRole}
          options={[
            { value: "patient", label: "Patient / athlete" },
            { value: "clinician", label: "Clinician / trainer" },
          ]}
        />
      </div>

      {role === "clinician" ? (
        <PlanEditor library={library} onChanged={refreshAll} />
      ) : run ? (
        <SessionRunner
          run={run}
          metrics={metrics}
          frame={frame}
          connection={connection}
          exercise={exercise}
          onClose={(saved) => {
            exercise("stop");
            setRun(null);
            if (saved) refreshAll();
          }}
        />
      ) : (
        <div className="grid">
          <div className="span-7">
            <TodayPanel
              slots={today.data}
              error={today.error}
              onStart={(slot) => {
                const spec = specFor(slot.item.exercise);
                if (spec) setRun(runFromItem(spec, slot.item, slot));
              }}
            />
          </div>
          <div className="span-5">
            <PracticePanel library={library} onStart={(spec) => setRun(runFromItem(spec, null, null))} />
          </div>
          <div className="span-12">
            <HistoryPanel sessions={sessions.data} adherence={adherence.data} />
          </div>
        </div>
      )}
    </>
  );
}

/* ----------------------------------------------------------------- today */

function TodayPanel({
  slots,
  error,
  onStart,
}: {
  slots: ScheduleSlot[] | null;
  error: string | null;
  onStart: (slot: ScheduleSlot) => void;
}) {
  const done = slots?.filter((s) => s.status === "done").length ?? 0;
  return (
    <Panel title="Today's plan" meta={slots?.length ? `${done} of ${slots.length} done` : undefined}>
      {error ? <Note>{error}</Note> : null}
      {!slots ? (
        <Empty>Loading…</Empty>
      ) : !slots.length ? (
        <Empty>Nothing scheduled today. Your clinician or trainer adds exercises under “Clinician / trainer”.</Empty>
      ) : (
        <ul className="slots">
          {slots.map((slot) => (
            <li key={`${slot.item.id}-${slot.time}`} className="slot" data-status={slot.status}>
              <span className="slot-time figure">{slot.time ?? "Any time"}</span>
              <div className="slot-main">
                <strong>{slot.item.exercise_name}</strong>
                <span className="muted">
                  {dose(slot.item)}, rest {slot.item.rest_s} s
                  {slot.item.side !== "auto" ? `, ${slot.item.side} side` : ""}
                </span>
                <span className="muted slot-by">
                  {slot.program.title}
                  {slot.program.prescriber ? ` · ${slot.program.prescriber_role ?? ""} ${slot.program.prescriber}` : ""}
                </span>
                {slot.item.instructions ? <span className="slot-note">“{slot.item.instructions}”</span> : null}
              </div>
              <Badge tone={STATUS_TONE[slot.status]}>{slot.status}</Badge>
              <button
                type="button"
                className={`btn btn--small ${slot.status === "done" ? "btn--quiet" : "btn--primary"}`}
                onClick={() => onStart(slot)}
              >
                {slot.status === "done" ? "Again" : "Start"}
              </button>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

function PracticePanel({ library, onStart }: { library: ExerciseSpec[]; onStart: (spec: ExerciseSpec) => void }) {
  const [key, setKey] = useState("");
  const spec = library.find((e) => e.key === key) ?? library[0];
  return (
    <Panel title="Free practice" meta="Not part of a plan">
      <div className="form-row">
        <label className="field field--grow">
          <span>Exercise</span>
          <ExerciseSelect library={library} value={spec?.key ?? ""} onChange={setKey} />
        </label>
        <button type="button" className="btn btn--primary" disabled={!spec} onClick={() => spec && onStart(spec)}>
          Start
        </button>
      </div>
      {spec ? (
        <>
          <p className="muted practice-setup">{spec.setup}</p>
          <ul className="cue-list">
            {spec.cues.map((cue) => (
              <li key={cue}>{cue}</li>
            ))}
          </ul>
        </>
      ) : null}
    </Panel>
  );
}

function ExerciseSelect({
  library,
  value,
  onChange,
}: {
  library: ExerciseSpec[];
  value: string;
  onChange: (key: string) => void;
}) {
  const groups = useMemo(() => {
    const map = new Map<string, ExerciseSpec[]>();
    for (const item of library) map.set(item.category, [...(map.get(item.category) ?? []), item]);
    return [...map.entries()];
  }, [library]);
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {groups.map(([category, items]) => (
        <optgroup key={category} label={category[0].toUpperCase() + category.slice(1)}>
          {items.map((item) => (
            <option key={item.key} value={item.key}>
              {item.name}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}

/* -------------------------------------------------------------- session */

type Phase = "baseline" | "work" | "rest" | "recovery" | "review";

interface SetResult {
  reps: number;
  partials: number;
  hold_s: number;
  avg_rom: number | null;
  best_rom: number | null;
  avg_form: number | null;
  rep_log: RepLog[];
}

const BASELINE_S = 15;
const RECOVERY_S = 60;

function SessionRunner({
  run,
  metrics,
  frame,
  connection,
  exercise,
  onClose,
}: {
  run: Run;
  metrics: Metrics | null;
  frame: ImageBitmap | null;
  connection: ConnectionState;
  exercise: ExerciseCommand;
  onClose: (saved: boolean) => void;
}) {
  const { spec } = run;
  const isHold = spec.kind === "hold";
  const [phase, setPhase] = useState<Phase>("baseline");
  const [setIndex, setSetIndex] = useState(0);
  const [clock, setClock] = useState(0);
  const [results, setResults] = useState<SetResult[]>([]);
  const [voice, setVoice] = useState(true);
  const [stoppedEarly, setStoppedEarly] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [rpe, setRpe] = useState(5);
  const [pain, setPain] = useState(0);
  const [notes, setNotes] = useState("");
  const startedAt = useRef(Date.now() / 1000);
  const hr = useRef<{ phase: Phase; t: number; bpm: number }[]>([]);
  const armed = useRef(false);
  const lastReps = useRef(0);
  const metricsRef = useRef(metrics);
  metricsRef.current = metrics;

  const live: ExerciseLive | null = metrics?.body?.session ?? null;
  const bpm = metrics?.bpm ?? null;
  const overLimit = run.max_hr !== null && bpm !== null && bpm > run.max_hr;

  // One tick per second: phase clock + heart-rate log.
  useEffect(() => {
    const timer = window.setInterval(() => {
      const value = metricsRef.current?.bpm;
      if (value != null) hr.current.push({ phase, t: Date.now() / 1000, bpm: value });
      setClock((c) => c + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [phase]);

  const enter = useCallback(
    (next: Phase) => {
      setClock(0);
      setPhase(next);
      if (next === "work") {
        armed.current = false;
        lastReps.current = 0;
        exercise("start", spec.key, run.side);
        if (voice) speak(`Set ${setIndex + 1}. Go.`);
      } else {
        exercise("stop");
      }
    },
    [exercise, spec.key, run.side, voice, setIndex],
  );

  const finishSet = useCallback(
    (session: ExerciseLive | null) => {
      const result: SetResult = {
        reps: session?.reps ?? 0,
        partials: session?.partials ?? 0,
        hold_s: session?.hold_s ?? 0,
        avg_rom: session?.avg_rom ?? null,
        best_rom: session?.best_rom ?? null,
        avg_form: session?.avg_form ?? null,
        rep_log: session?.rep_log ?? [],
      };
      setResults((previous) => [...previous, result]);
      const last = setIndex + 1 >= run.sets;
      if (last) {
        if (voice) speak("Great work. Sit down and breathe for one minute.");
        enter("recovery");
      } else {
        if (voice) speak(`Set done. Rest ${run.rest_s} seconds.`);
        setSetIndex((i) => i + 1);
        enter("rest");
      }
    },
    [enter, run.sets, run.rest_s, setIndex, voice],
  );

  // Arm on a fresh counter, then watch for the target.
  useEffect(() => {
    if (phase !== "work" || !live) return;
    if (!armed.current) {
      if (live.reps === 0 && live.hold_s < 0.5) armed.current = true;
      return;
    }
    if (!isHold && live.reps > lastReps.current) {
      lastReps.current = live.reps;
      if (voice) speak(String(live.reps));
    }
    if ((!isHold && live.reps >= run.reps) || (isHold && live.hold_s >= run.hold_s)) finishSet(live);
  }, [phase, live, isHold, run.reps, run.hold_s, finishSet, voice]);

  // Timed phases.
  useEffect(() => {
    if (phase === "baseline" && clock >= BASELINE_S) enter("work");
    if (phase === "rest" && clock >= run.rest_s && !overLimit) enter("work");
    if (phase === "recovery" && clock >= RECOVERY_S) enter("review");
  }, [phase, clock, run.rest_s, overLimit, enter]);

  useEffect(() => () => exercise("stop"), [exercise]);

  const stopEarly = () => {
    if (phase === "work") {
      setResults((previous) => [
        ...previous,
        {
          reps: live?.reps ?? 0,
          partials: live?.partials ?? 0,
          hold_s: live?.hold_s ?? 0,
          avg_rom: live?.avg_rom ?? null,
          best_rom: live?.best_rom ?? null,
          avg_form: live?.avg_form ?? null,
          rep_log: live?.rep_log ?? [],
        },
      ]);
    }
    setStoppedEarly(true);
    enter("review");
  };

  const summary = useMemo(() => {
    const reps = results.reduce((sum, r) => sum + r.reps, 0);
    const logs = results.flatMap((r) => r.rep_log);
    const roms = logs.map((r) => r.rom);
    const forms = logs.map((r) => r.form_score);
    const by = (p: Phase) => hr.current.filter((s) => s.phase === p).map((s) => s.bpm);
    const recovery = hr.current.filter((s) => s.phase === "recovery");
    const firstT = recovery[0]?.t ?? 0;
    const hrPost = median(recovery.filter((s) => s.t - firstT <= 10).map((s) => s.bpm));
    const hrEnd = median(recovery.filter((s) => s.t - firstT >= 45).map((s) => s.bpm));
    const all = hr.current.map((s) => s.bpm);
    return {
      reps,
      partials: results.reduce((sum, r) => sum + r.partials, 0),
      hold: results.reduce((sum, r) => sum + r.hold_s, 0),
      avgRom: roms.length ? roms.reduce((a, b) => a + b, 0) / roms.length : null,
      bestRom: roms.length ? Math.max(...roms) : null,
      avgForm: forms.length ? forms.reduce((a, b) => a + b, 0) / forms.length : null,
      hrPre: median(by("baseline")),
      hrPeak: all.length ? Math.max(...all) : null,
      hrPost,
      hrRecovery: hrPost !== null && hrEnd !== null ? hrPost - hrEnd : null,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [results, phase]);

  const save = async () => {
    setSaving(true);
    setSaveError(null);
    const record: ExerciseSession = {
      program_id: run.program_id,
      item_id: run.item_id,
      exercise: spec.key,
      slot_date: run.slot_date,
      slot_time: run.slot_time,
      started_at: startedAt.current,
      ended_at: Date.now() / 1000,
      sets_done: results.length,
      reps_done: summary.reps,
      reps_target: isHold ? 0 : run.sets * run.reps,
      hold_s: isHold ? summary.hold : null,
      partials: summary.partials,
      avg_form: summary.avgForm,
      avg_rom: summary.avgRom,
      best_rom: summary.bestRom,
      hr_pre: summary.hrPre,
      hr_peak: summary.hrPeak,
      hr_post: summary.hrPost,
      hr_recovery: summary.hrRecovery,
      rpe,
      pain,
      notes: notes.trim() || null,
      completed: !stoppedEarly && results.length >= run.sets,
      detail: { sets: results, side: run.side },
    };
    try {
      await athletics.saveSession(record);
      onClose(true);
    } catch (exc) {
      setSaveError(exc instanceof Error ? exc.message : "Could not save.");
      setSaving(false);
    }
  };

  const target = isHold ? `${run.hold_s} s` : `${run.reps}`;
  const progress = isHold ? (live?.hold_s ?? 0) / Math.max(run.hold_s, 1) : (live?.reps ?? 0) / Math.max(run.reps, 1);
  const remaining =
    phase === "baseline" ? BASELINE_S - clock : phase === "rest" ? run.rest_s - clock : phase === "recovery" ? RECOVERY_S - clock : null;

  return (
    <div className="grid">
      <div className="span-8 stage-cell">
        <VideoStage frame={frame} metrics={metrics} connection={connection} mode="body" />
      </div>
      <div className="span-4 vitals">
        <section className="vital ath-hud" aria-live="polite">
          <div className="vital-head">
            <h2 className="vital-name">{spec.name}</h2>
            <Badge tone={phase === "work" ? "ok" : phase === "rest" ? "info" : "idle"}>
              {phase === "work" ? `Set ${setIndex + 1} of ${run.sets}` : phase}
            </Badge>
          </div>
          {phase === "work" ? (
            <>
              <p className="rep-count figure">
                {isHold ? Math.floor(live?.hold_s ?? 0) : (live?.reps ?? 0)}
                <span className="rep-target">/ {target}</span>
              </p>
              <Meter value={progress} tone="ok" label="Set progress" />
              <ul className="cue-list cue-list--live">
                {(live?.cues.length ? live.cues : spec.cues.slice(0, 1)).map((cue) => (
                  <li key={cue}>{cue}</li>
                ))}
              </ul>
              <Stats columns={3}>
                <Stat label="Angle" value={num(live?.angle)} unit="°" />
                <Stat
                  label="Last ROM"
                  value={num(live?.last_rep?.rom)}
                  unit="°"
                  tone={
                    run.target_rom !== null && live?.last_rep
                      ? live.last_rep.rom >= run.target_rom
                        ? "ok"
                        : "warn"
                      : undefined
                  }
                />
                <Stat
                  label="Form"
                  value={num(live?.last_rep?.form_score)}
                  unit="%"
                  tone={live?.last_rep ? (live.last_rep.form_score >= 75 ? "ok" : "warn") : undefined}
                />
              </Stats>
              {live?.partials ? <Note>{live.partials} partial rep(s) not counted.</Note> : null}
              <div className="dialog-actions">
                <button type="button" className="btn" onClick={() => finishSet(live)}>
                  End set
                </button>
              </div>
            </>
          ) : phase === "review" ? (
            <p className="muted">Session finished. Review and save below.</p>
          ) : (
            <>
              <p className="rep-count figure">{Math.max(0, remaining ?? 0)}<span className="rep-target">s</span></p>
              <p className="vital-state">
                {phase === "baseline"
                  ? "Sit still facing the camera — measuring resting heart rate"
                  : phase === "rest"
                    ? overLimit
                      ? `Heart rate above your limit of ${run.max_hr}. Keep resting.`
                      : `Rest. Next: set ${setIndex + 1} of ${run.sets}`
                    : "Sit still facing the camera — measuring heart-rate recovery"}
              </p>
              <div className="dialog-actions">
                <button
                  type="button"
                  className="btn"
                  onClick={() => enter(phase === "baseline" || phase === "rest" ? "work" : "review")}
                >
                  Skip
                </button>
              </div>
            </>
          )}
        </section>
        <section className="vital" data-alert={overLimit ? "true" : undefined}>
          <div className="vital-head">
            <h2 className="vital-name">Heart rate</h2>
            <span className="vital-unit">BPM</span>
          </div>
          <p className="vital-value vital-value--small">{bpm === null ? "– –" : Math.round(bpm)}</p>
          <p className="vital-state">
            {overLimit
              ? "Above the prescribed maximum — slow down"
              : bpm === null
                ? phase === "work"
                  ? "Read between sets (movement blocks the camera pulse)"
                  : "Face the camera and keep still"
                : run.max_hr
                  ? `Limit ${run.max_hr} BPM`
                  : "Camera estimate"}
          </p>
        </section>
        <div className="ath-controls">
          <label className="toggle">
            <input type="checkbox" checked={voice} onChange={(e) => setVoice(e.target.checked)} />
            <span>Voice count</span>
          </label>
          {phase !== "review" ? (
            <button type="button" className="btn btn--quiet" onClick={stopEarly}>
              Stop — pain or unwell
            </button>
          ) : null}
          <button type="button" className="btn btn--quiet" onClick={() => onClose(false)}>
            Discard
          </button>
        </div>
      </div>

      {run.instructions || phase === "baseline" ? (
        <div className="span-12">
          <Panel title="Before you start">
            <p>{spec.setup}</p>
            {run.instructions ? <p className="slot-note">From your clinician: “{run.instructions}”</p> : null}
            <ul className="cue-list">
              {spec.cues.map((cue) => (
                <li key={cue}>{cue}</li>
              ))}
            </ul>
          </Panel>
        </div>
      ) : null}

      {phase === "review" ? (
        <div className="span-12">
          <Panel title="Session summary" meta={stoppedEarly ? "Stopped early" : "Completed"}>
            <Stats columns={4}>
              <Stat label={isHold ? "Time held" : "Reps"} value={isHold ? Math.round(summary.hold) : summary.reps} unit={isHold ? "s" : `/ ${run.sets * run.reps}`} />
              <Stat label="Avg form" value={num(summary.avgForm)} unit="%" />
              <Stat label="Avg range of motion" value={num(summary.avgRom)} unit="°" />
              <Stat label="Best ROM" value={num(summary.bestRom)} unit="°" />
              <Stat label="Resting HR" value={num(summary.hrPre)} unit="BPM" />
              <Stat label="Peak HR" value={num(summary.hrPeak)} unit="BPM" />
              <Stat label="HR after" value={num(summary.hrPost)} unit="BPM" />
              <Stat label="1-min recovery" value={num(summary.hrRecovery)} unit="BPM" />
            </Stats>
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Set</th>
                  <th scope="col">{isHold ? "Held" : "Reps"}</th>
                  <th scope="col">Partial</th>
                  <th scope="col">Avg ROM</th>
                  <th scope="col">Form</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i}>
                    <th scope="row">{i + 1}</th>
                    <td className="figure">{isHold ? `${Math.round(r.hold_s)} s` : r.reps}</td>
                    <td className="figure">{r.partials}</td>
                    <td className="figure">{num(r.avg_rom)}°</td>
                    <td className="figure">{num(r.avg_form)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="form-grid">
              <label className="field">
                <span>Effort (RPE 0 = rest, 10 = maximal): {rpe}</span>
                <input type="range" min={0} max={10} value={rpe} onChange={(e) => setRpe(Number(e.target.value))} />
              </label>
              <label className="field">
                <span>Pain (0 = none, 10 = worst): {pain}</span>
                <input type="range" min={0} max={10} value={pain} onChange={(e) => setPain(Number(e.target.value))} />
              </label>
            </div>
            <label className="field">
              <span>Notes for your clinician</span>
              <input value={notes} maxLength={500} onChange={(e) => setNotes(e.target.value)} placeholder="Optional" />
            </label>
            {pain >= 7 ? <Note>Pain of 7 or more is flagged to your clinician. Stop and contact them if it persists.</Note> : null}
            {saveError ? <Note>{saveError}</Note> : null}
            <div className="dialog-actions">
              <button type="button" className="btn btn--primary" disabled={saving} onClick={() => void save()}>
                {saving ? "Saving…" : "Save session"}
              </button>
            </div>
          </Panel>
        </div>
      ) : null}
    </div>
  );
}

/* --------------------------------------------------------------- history */

function HistoryPanel({ sessions, adherence }: { sessions: ExerciseSession[] | null; adherence: AdherenceDay[] | null }) {
  const scheduled = adherence?.reduce((s, d) => s + d.scheduled, 0) ?? 0;
  const done = adherence?.reduce((s, d) => s + d.done, 0) ?? 0;
  return (
    <Panel title="Progress" meta={scheduled ? `Adherence ${Math.round((done / scheduled) * 100)}% over 14 days` : undefined}>
      {adherence?.length ? (
        <div className="adherence" role="img" aria-label="Daily adherence, last 14 days">
          {adherence.map((day) => {
            const ratio = day.scheduled ? day.done / day.scheduled : 0;
            return (
              <div key={day.date} className="adherence-day" title={`${day.date}: ${day.done}/${day.scheduled}`}>
                <span
                  className="adherence-bar"
                  data-empty={day.scheduled === 0 ? "true" : undefined}
                  style={{ height: `${Math.max(6, ratio * 100)}%` }}
                  data-tone={ratio >= 1 ? "ok" : ratio > 0 ? "warn" : "bad"}
                />
                <span className="adherence-label">{new Date(day.date).toLocaleDateString(undefined, { weekday: "narrow" })}</span>
              </div>
            );
          })}
        </div>
      ) : null}
      {!sessions?.length ? (
        <Empty>Completed sessions appear here with reps, form, range of motion and heart-rate recovery.</Empty>
      ) : (
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">When</th>
                <th scope="col">Exercise</th>
                <th scope="col">Done</th>
                <th scope="col">Form</th>
                <th scope="col">ROM</th>
                <th scope="col">HR rest / peak</th>
                <th scope="col">1-min recovery</th>
                <th scope="col">Effort</th>
                <th scope="col">Pain</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.id}>
                  <th scope="row">
                    {new Date(s.started_at * 1000).toLocaleString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </th>
                  <td>
                    {s.exercise_name}
                    {!s.completed ? <Badge tone="warn">stopped</Badge> : null}
                  </td>
                  <td className="figure">{s.hold_s != null ? `${Math.round(s.hold_s)} s` : `${s.reps_done}/${s.reps_target}`}</td>
                  <td className="figure">{num(s.avg_form)}%</td>
                  <td className="figure">{num(s.avg_rom)}°</td>
                  <td className="figure">
                    {num(s.hr_pre)} / {num(s.hr_peak)}
                  </td>
                  <td className="figure">{num(s.hr_recovery)}</td>
                  <td className="figure">{num(s.rpe)}</td>
                  <td className="figure" data-tone={(s.pain ?? 0) >= 7 ? "bad" : undefined}>
                    {num(s.pain)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <Note>Heart-rate recovery is the drop in the first minute after exercise; a bigger drop usually means better fitness.</Note>
    </Panel>
  );
}

/* ------------------------------------------------------------ plan editor */

function PlanEditor({ library, onChanged }: { library: ExerciseSpec[]; onChanged: () => void }) {
  const [pinSet, setPinSet] = useState<boolean | null>(null);
  const [unlocked, setUnlocked] = useState(false);
  const [pin, setPin] = useState("");
  const [error, setError] = useState<string | null>(null);
  const programs = usePoll(() => athletics.programs(), 60000);

  useEffect(() => {
    athletics
      .pinStatus()
      .then((status) => {
        setPinSet(status.set);
        if (!status.set) setUnlocked(true);
      })
      .catch(() => setPinSet(false));
  }, []);

  const act = async (work: () => Promise<unknown>) => {
    setError(null);
    try {
      await work();
      await programs.refresh();
      onChanged();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Request failed.");
    }
  };

  if (pinSet && !unlocked) {
    return (
      <Panel title="Clinician access">
        <form
          className="form-row"
          onSubmit={(event) => {
            event.preventDefault();
            athletics
              .verifyPin(pin)
              .then((r) => {
                if (r.ok) {
                  setClinicianPin(pin);
                  setUnlocked(true);
                } else setError("Wrong PIN.");
              })
              .catch(() => setError("Could not check the PIN."));
          }}
        >
          <label className="field">
            <span>PIN</span>
            <input type="password" inputMode="numeric" value={pin} onChange={(e) => setPin(e.target.value)} autoFocus />
          </label>
          <button type="submit" className="btn btn--primary">
            Unlock
          </button>
        </form>
        {error ? <Note>{error}</Note> : null}
      </Panel>
    );
  }

  return (
    <div className="grid">
      <div className="span-8">
        {error ? <Note>{error}</Note> : null}
        {!programs.data?.length ? (
          <Panel title="Programs">
            <Empty>No programs yet. Create one on the right, then add exercises to it.</Empty>
          </Panel>
        ) : (
          programs.data.map((program) => (
            <ProgramCard
              key={program.id}
              program={program}
              library={library}
              onAct={act}
            />
          ))
        )}
      </div>
      <div className="span-4">
        <NewProgramPanel onCreate={(data) => act(() => athletics.createProgram(data))} />
        <PinPanel
          pinSet={!!pinSet}
          onChange={(next) =>
            act(async () => {
              const result = await athletics.changePin(next, pin);
              setPinSet(result.set);
              setPin(next ?? "");
              setClinicianPin(next ?? "");
            })
          }
        />
      </div>
    </div>
  );
}

function NewProgramPanel({ onCreate }: { onCreate: (data: Partial<Program>) => Promise<void> }) {
  const [form, setForm] = useState({ title: "", patient: "", prescriber: "", prescriber_role: ROLES[0], goal: "", end_date: "" });
  const set = (key: keyof typeof form) => (e: { target: { value: string } }) => setForm({ ...form, [key]: e.target.value });
  return (
    <Panel title="New program">
      <form
        className="stack"
        onSubmit={(event) => {
          event.preventDefault();
          if (!form.title.trim()) return;
          void onCreate({
            title: form.title.trim(),
            patient: form.patient || null,
            prescriber: form.prescriber || null,
            prescriber_role: form.prescriber_role,
            goal: form.goal || null,
            end_date: form.end_date || null,
          }).then(() => setForm({ ...form, title: "", patient: "", goal: "", end_date: "" }));
        }}
      >
        <label className="field">
          <span>Title</span>
          <input value={form.title} onChange={set("title")} placeholder="e.g. Post-op knee, weeks 2–6" required maxLength={120} />
        </label>
        <label className="field">
          <span>Patient / athlete</span>
          <input value={form.patient} onChange={set("patient")} maxLength={80} />
        </label>
        <div className="form-grid">
          <label className="field">
            <span>Prescribed by</span>
            <input value={form.prescriber} onChange={set("prescriber")} maxLength={80} placeholder="Name" />
          </label>
          <label className="field">
            <span>Role</span>
            <select value={form.prescriber_role} onChange={set("prescriber_role")}>
              {ROLES.map((role) => (
                <option key={role}>{role}</option>
              ))}
            </select>
          </label>
        </div>
        <label className="field">
          <span>Goal</span>
          <input value={form.goal} onChange={set("goal")} maxLength={300} placeholder="e.g. Knee flexion to 120°" />
        </label>
        <label className="field">
          <span>Ends on (optional)</span>
          <input type="date" value={form.end_date} onChange={set("end_date")} />
        </label>
        <button type="submit" className="btn btn--primary">
          Create program
        </button>
      </form>
    </Panel>
  );
}

function ProgramCard({
  program,
  library,
  onAct,
}: {
  program: Program;
  library: ExerciseSpec[];
  onAct: (work: () => Promise<unknown>) => Promise<void>;
}) {
  const [adding, setAdding] = useState(false);
  return (
    <Panel
      title={program.title}
      meta={[program.patient, program.prescriber ? `${program.prescriber_role ?? ""} ${program.prescriber}` : null]
        .filter(Boolean)
        .join(" · ")}
      actions={
        <>
          <label className="toggle toggle--inline">
            <input
              type="checkbox"
              checked={program.active}
              onChange={(e) => void onAct(() => athletics.updateProgram(program.id, { ...program, active: e.target.checked }))}
            />
            <span>Active</span>
          </label>
          <button
            type="button"
            className="link link--danger"
            onClick={() => {
              if (window.confirm(`Delete “${program.title}” and its exercises?`)) void onAct(() => athletics.deleteProgram(program.id));
            }}
          >
            Delete
          </button>
        </>
      }
    >
      {program.goal ? <p className="muted">Goal: {program.goal}</p> : null}
      {program.items.length ? (
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">Exercise</th>
                <th scope="col">Dose</th>
                <th scope="col">Rest</th>
                <th scope="col">Days</th>
                <th scope="col">Times</th>
                <th scope="col">Limits</th>
                <th scope="col" aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {program.items.map((item) => (
                <tr key={item.id}>
                  <th scope="row">
                    {item.exercise_name}
                    {item.side !== "auto" ? <span className="muted"> ({item.side})</span> : null}
                    {item.instructions ? <div className="muted small">{item.instructions}</div> : null}
                  </th>
                  <td className="figure">{dose(item)}</td>
                  <td className="figure">{item.rest_s} s</td>
                  <td>{item.days === "1111111" ? "Daily" : DAYS.filter((_, i) => item.days[i] === "1").join(" ")}</td>
                  <td className="figure">{item.times.join(", ") || "Any"}</td>
                  <td className="figure">
                    {[item.target_rom ? `ROM ${item.target_rom}°` : null, item.max_hr ? `HR ≤ ${item.max_hr}` : null]
                      .filter(Boolean)
                      .join(", ") || "–"}
                  </td>
                  <td>
                    <button type="button" className="link link--danger" onClick={() => void onAct(() => athletics.deleteItem(item.id))}>
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>No exercises yet.</Empty>
      )}
      {adding ? (
        <ItemForm
          library={library}
          onCancel={() => setAdding(false)}
          onSave={(data) => onAct(() => athletics.addItem(program.id, data)).then(() => setAdding(false))}
        />
      ) : (
        <div className="panel-footer">
          <button type="button" className="btn" onClick={() => setAdding(true)}>
            Add exercise
          </button>
        </div>
      )}
    </Panel>
  );
}

function ItemForm({
  library,
  onSave,
  onCancel,
}: {
  library: ExerciseSpec[];
  onSave: (data: Record<string, unknown>) => Promise<void>;
  onCancel: () => void;
}) {
  const [key, setKey] = useState(library[0]?.key ?? "");
  const spec = library.find((e) => e.key === key);
  const [sets, setSets] = useState(3);
  const [reps, setReps] = useState(10);
  const [hold, setHold] = useState(30);
  const [rest, setRest] = useState(60);
  const [side, setSide] = useState("auto");
  const [days, setDays] = useState<boolean[]>(Array(7).fill(true));
  const [times, setTimes] = useState<string[]>(["09:00"]);
  const [rom, setRom] = useState("");
  const [maxHr, setMaxHr] = useState("");
  const [instructions, setInstructions] = useState("");

  useEffect(() => {
    if (!spec) return;
    setSets(spec.default_sets);
    setReps(spec.default_reps);
    setHold(spec.default_hold_s);
    setRest(spec.default_rest_s);
  }, [spec]);

  return (
    <form
      className="item-form"
      onSubmit={(event) => {
        event.preventDefault();
        void onSave({
          exercise: key,
          sets,
          reps: spec?.kind === "hold" ? 1 : reps,
          hold_s: spec?.kind === "hold" ? hold : 0,
          rest_s: rest,
          side,
          days: days.flatMap((on, i) => (on ? [i] : [])),
          times: times.filter(Boolean),
          target_rom: rom ? Number(rom) : null,
          max_hr: maxHr ? Number(maxHr) : null,
          instructions: instructions.trim() || null,
        });
      }}
    >
      <div className="form-row">
        <label className="field field--grow">
          <span>Exercise</span>
          <ExerciseSelect library={library} value={key} onChange={setKey} />
        </label>
        <label className="field">
          <span>Sets</span>
          <input type="number" min={1} max={20} value={sets} onChange={(e) => setSets(Number(e.target.value))} />
        </label>
        {spec?.kind === "hold" ? (
          <label className="field">
            <span>Hold (s)</span>
            <input type="number" min={5} max={600} value={hold} onChange={(e) => setHold(Number(e.target.value))} />
          </label>
        ) : (
          <label className="field">
            <span>Reps</span>
            <input type="number" min={1} max={200} value={reps} onChange={(e) => setReps(Number(e.target.value))} />
          </label>
        )}
        <label className="field">
          <span>Rest (s)</span>
          <input type="number" min={0} max={900} value={rest} onChange={(e) => setRest(Number(e.target.value))} />
        </label>
        <label className="field">
          <span>Side</span>
          <select value={side} onChange={(e) => setSide(e.target.value)}>
            <option value="auto">Auto</option>
            <option value="left">Left</option>
            <option value="right">Right</option>
            <option value="both">Both</option>
          </select>
        </label>
      </div>
      <fieldset className="day-picker">
        <legend>Days</legend>
        {DAYS.map((day, i) => (
          <label key={day} className="chip" data-on={days[i] ? "true" : undefined}>
            <input
              type="checkbox"
              checked={days[i]}
              onChange={(e) => setDays(days.map((v, j) => (j === i ? e.target.checked : v)))}
            />
            {day}
          </label>
        ))}
      </fieldset>
      <div className="form-row">
        {times.map((time, i) => (
          <label key={i} className="field">
            <span>Time {i + 1}</span>
            <input type="time" value={time} onChange={(e) => setTimes(times.map((t, j) => (j === i ? e.target.value : t)))} />
          </label>
        ))}
        {times.length < 4 ? (
          <button type="button" className="btn btn--quiet btn--small" onClick={() => setTimes([...times, "18:00"])}>
            + time
          </button>
        ) : null}
        {times.length > 1 ? (
          <button type="button" className="btn btn--quiet btn--small" onClick={() => setTimes(times.slice(0, -1))}>
            − time
          </button>
        ) : null}
      </div>
      <div className="form-row">
        <label className="field">
          <span>Target ROM (°)</span>
          <input type="number" min={0} max={180} value={rom} onChange={(e) => setRom(e.target.value)} placeholder="Optional" />
        </label>
        <label className="field">
          <span>Max HR (BPM)</span>
          <input type="number" min={60} max={220} value={maxHr} onChange={(e) => setMaxHr(e.target.value)} placeholder="Optional" />
        </label>
        <label className="field field--grow">
          <span>Instructions for the patient</span>
          <input value={instructions} maxLength={500} onChange={(e) => setInstructions(e.target.value)} placeholder="e.g. Slow on the way down" />
        </label>
      </div>
      {spec ? <Note>{spec.setup}</Note> : null}
      <div className="dialog-actions">
        <button type="button" className="btn btn--quiet" onClick={onCancel}>
          Cancel
        </button>
        <button type="submit" className="btn btn--primary">
          Add to program
        </button>
      </div>
    </form>
  );
}

function PinPanel({ pinSet, onChange }: { pinSet: boolean; onChange: (pin: string | null) => Promise<void> }) {
  const [next, setNext] = useState("");
  return (
    <Panel title="Clinician PIN" meta={pinSet ? "On" : "Off"}>
      <p className="muted">
        {pinSet
          ? "Plan changes need the PIN. Patients can still run sessions."
          : "Set a PIN so patients cannot change the prescription."}
      </p>
      <form
        className="form-row"
        onSubmit={(event) => {
          event.preventDefault();
          if (next.length >= 4) void onChange(next).then(() => setNext(""));
        }}
      >
        <label className="field field--grow">
          <span>{pinSet ? "New PIN" : "PIN (4+ digits)"}</span>
          <input type="password" inputMode="numeric" minLength={4} value={next} onChange={(e) => setNext(e.target.value)} />
        </label>
        <button type="submit" className="btn">
          {pinSet ? "Change" : "Set"}
        </button>
        {pinSet ? (
          <button type="button" className="btn btn--quiet" onClick={() => void onChange(null)}>
            Remove
          </button>
        ) : null}
      </form>
    </Panel>
  );
}
