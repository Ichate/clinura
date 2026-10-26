import { type FormEvent, useEffect, useRef, useState } from "react";

import { api, type AssistantStatus } from "../lib/api";
import { duration, num } from "../lib/format";
import type { ChatMessage, ClinicalEvent, HandoverResult, TelemetrySummary, Tone } from "../lib/types";
import { Chart } from "./Chart";
import { CopyIcon, MicIcon, PlusIcon, SendIcon } from "./Icons";
import { RichText } from "./RichText";
import { Badge, Empty, Note, Panel, Stat, Stats } from "./ui";

const SEVERITY_TONE: Record<string, Tone> = { info: "info", warning: "warn", critical: "bad" };

export function OverviewPanel({ summary, hours }: { summary: TelemetrySummary | null; hours: number }) {
  const hr = summary?.heart_rate_bpm;
  const series = summary?.series ?? [];
  const now = Date.now() / 1000;
  const x = series.map((bin) => (new Date(bin.from.replace(" ", "T")).getTime() / 1000 - now) / 3600);
  return (
    <Panel title="Session overview" meta={`Last ${hours} h`}>
      {summary && summary.window.seconds_recorded > 0 ? (
        <>
          <Stats columns={3}>
            <Stat
              label="Heart rate"
              value={hr ? `${hr.mean.toFixed(0)} ± ${hr.sd.toFixed(0)}` : "–"}
              unit="BPM"
            />
            <Stat label="RMSSD" value={num(summary.rmssd_ms?.mean)} unit="ms" />
            <Stat label="Respiration" value={num(summary.respiration_bpm?.mean)} unit="/min" />
            <Stat
              label="Range"
              value={
                summary.heart_rate_lowest && summary.heart_rate_peak
                  ? `${summary.heart_rate_lowest.bpm.toFixed(0)}–${summary.heart_rate_peak.bpm.toFixed(0)}`
                  : "–"
              }
              unit="BPM"
            />
            <Stat label="In view" value={summary.coverage.patient_in_view_pct.toFixed(0)} unit="%" />
            <Stat label="Readable" value={summary.coverage.confirmed_hr_pct.toFixed(0)} unit="%" />
          </Stats>
          <Chart
            label="Average heart rate over the session"
            x={x}
            y={series.map((bin) => bin.hr_mean)}
            xRange={[-hours, 0]}
            color="--hr"
            points
            height={120}
            showY
            xTick={(t) => (t === 0 ? "now" : `${t.toFixed(hours <= 1 ? 2 : 0)} h`)}
            yTick={(t) => t.toFixed(0)}
            empty="Not enough data for a trend"
          />
        </>
      ) : (
        <Empty>Nothing recorded in this window yet. Telemetry is logged once a second while the camera runs.</Empty>
      )}
    </Panel>
  );
}

export function HandoverPanel({
  hours,
  bed,
  patient,
  onBed,
  onPatient,
  assistant,
}: {
  hours: number;
  bed: string;
  patient: string;
  onBed: (value: string) => void;
  onPatient: (value: string) => void;
  assistant: AssistantStatus | null;
}) {
  const [result, setResult] = useState<HandoverResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.handover(hours, bed.trim(), patient.trim()));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not generate the handover.");
    } finally {
      setBusy(false);
    }
  };

  const copy = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setError("Copy failed. Select the text and copy it manually.");
    }
  };

  return (
    <Panel
      title="Shift handover"
      meta={assistant ? (assistant.configured ? `Written by ${assistant.model}` : "Template mode") : undefined}
      className="handover"
    >
      <div className="form-row">
        <label className="field">
          <span>Bed</span>
          <input value={bed} maxLength={40} placeholder="04" onChange={(e) => onBed(e.target.value)} />
        </label>
        <label className="field field--grow">
          <span>Patient label</span>
          <input
            value={patient}
            maxLength={80}
            placeholder="Initials or ID, avoid full names"
            onChange={(e) => onPatient(e.target.value)}
          />
        </label>
        <button type="button" className="btn btn--primary" onClick={generate} disabled={busy}>
          {busy ? "Writing…" : `Write ${hours} h handover`}
        </button>
      </div>
      {error ? <p className="alert" role="alert">{error}</p> : null}
      {result ? (
        <article className="document" aria-live="polite">
          <header className="document-head">
            <span>
              {new Date(result.generated_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })},{" "}
              {result.generated_by}
            </span>
            <button type="button" className="btn btn--quiet btn--small" onClick={copy}>
              <CopyIcon /> {copied ? "Copied" : "Copy"}
            </button>
          </header>
          <RichText text={result.text} />
        </article>
      ) : (
        <Empty>
          Summarises vitals, detected events and staff notes from the selected window into a handover
          note you can paste into the record.
        </Empty>
      )}
      {assistant && !assistant.configured ? (
        <Note>
          Without a DeepSeek key the note is built from a fixed template. Add DEEPSEEK_API_KEY to .env
          for a written narrative.
        </Note>
      ) : (
        <Note>Telemetry summaries, not video, are sent to DeepSeek to write the note. Review before use.</Note>
      )}
    </Panel>
  );
}

type Recognition = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onend: (() => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  start: () => void;
  stop: () => void;
};

function speechRecognition(): (new () => Recognition) | null {
  const w = window as unknown as Record<string, unknown>;
  return (w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null) as (new () => Recognition) | null;
}

const SUGGESTIONS = [
  "Were there any heart-rate spikes while the eyes were closed?",
  "Summarise the last hour in two sentences.",
  "When was the patient out of view, and for how long?",
];

export function CopilotPanel({ hours, assistant }: { hours: number; assistant: AssistantStatus | null }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef<Recognition | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const Speech = speechRecognition();

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  useEffect(() => () => recognitionRef.current?.stop(), []);

  const send = async (question: string) => {
    const text = question.trim();
    if (!text || busy) return;
    const history = messages;
    setMessages([...history, { role: "user", content: text }]);
    setDraft("");
    setBusy(true);
    try {
      const response = await api.ask(text, hours, history);
      setMessages((current) => [...current, { role: "assistant", content: response.answer }]);
    } catch (exc) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: exc instanceof Error ? exc.message : "No answer.", error: true },
      ]);
    } finally {
      setBusy(false);
    }
  };

  const toggleVoice = () => {
    if (!Speech) return;
    if (listening) {
      recognitionRef.current?.stop();
      return;
    }
    const recognition = new Speech();
    recognition.lang = navigator.language || "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0]?.transcript ?? "")
        .join("");
      setDraft(transcript);
    };
    recognition.onend = () => setListening(false);
    recognition.onerror = () => setListening(false);
    recognitionRef.current = recognition;
    setListening(true);
    recognition.start();
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void send(draft);
  };

  const disabled = assistant !== null && !assistant.configured;

  return (
    <Panel title="Ask Clinura" meta={`Uses the last ${hours} h`} className="copilot">
      <div className="chat" ref={logRef} aria-live="polite">
        {messages.length === 0 ? (
          <div className="chat-intro">
            <p>Ask about the patient's monitored state. Answers cite times and values from the log.</p>
            <div className="suggestions">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  className="suggestion"
                  disabled={disabled || busy}
                  onClick={() => void send(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className="msg" data-role={message.role} data-error={message.error ? "true" : undefined}>
              {message.role === "assistant" && !message.error ? <RichText text={message.content} /> : <p>{message.content}</p>}
            </div>
          ))
        )}
        {busy ? (
          <div className="msg" data-role="assistant">
            <p className="typing">Reading the log…</p>
          </div>
        ) : null}
      </div>
      {disabled ? (
        <p className="alert">Add DEEPSEEK_API_KEY to the .env file and restart Clinura to enable questions.</p>
      ) : null}
      <form className="composer" onSubmit={submit}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={listening ? "Listening…" : "Ask about heart rate, events, rest periods…"}
          maxLength={1000}
          disabled={disabled}
          aria-label="Question"
        />
        {Speech ? (
          <button
            type="button"
            className="btn btn--icon"
            aria-pressed={listening}
            aria-label={listening ? "Stop voice input" : "Ask by voice"}
            onClick={toggleVoice}
            disabled={disabled}
          >
            <MicIcon />
          </button>
        ) : null}
        <button type="submit" className="btn btn--primary btn--icon" aria-label="Send question" disabled={disabled || busy || !draft.trim()}>
          <SendIcon />
        </button>
      </form>
    </Panel>
  );
}

const QUICK_NOTES = ["Medication given", "Nurse call", "Repositioned", "Patient asleep"];

export function EventLogPanel({
  events,
  error,
  onAdded,
}: {
  events: ClinicalEvent[] | null;
  error: string | null;
  onAdded: () => void;
}) {
  const [title, setTitle] = useState("");
  const [detail, setDetail] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const save = async (noteTitle: string, noteDetail = "") => {
    if (!noteTitle.trim()) return;
    setSaving(true);
    setSaveError(null);
    try {
      await api.addNote(noteTitle.trim(), noteDetail.trim());
      setTitle("");
      setDetail("");
      onAdded();
    } catch (exc) {
      setSaveError(exc instanceof Error ? exc.message : "Could not save the note.");
    } finally {
      setSaving(false);
    }
  };

  const sorted = [...(events ?? [])].reverse();

  return (
    <Panel title="Event log" meta={events ? `${events.length} entries` : undefined} className="events">
      <form
        className="note-form"
        onSubmit={(e) => {
          e.preventDefault();
          void save(title, detail);
        }}
      >
        <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Add a note, e.g. Paracetamol 1 g" maxLength={120} aria-label="Note" />
        <input value={detail} onChange={(e) => setDetail(e.target.value)} placeholder="Detail (optional)" maxLength={500} aria-label="Note detail" />
        <button type="submit" className="btn" disabled={saving || !title.trim()}>
          <PlusIcon /> Log
        </button>
      </form>
      <div className="quick-notes">
        {QUICK_NOTES.map((note) => (
          <button key={note} type="button" className="chip" disabled={saving} onClick={() => void save(note)}>
            {note}
          </button>
        ))}
      </div>
      {saveError || error ? <p className="alert" role="alert">{saveError ?? error}</p> : null}
      {sorted.length ? (
        <ol className="timeline">
          {sorted.map((event) => (
            <li key={event.id} data-severity={event.severity} data-source={event.source}>
              <time dateTime={event.time}>{event.clock.slice(0, 5)}</time>
              <div>
                <p className="timeline-title">
                  {event.title}
                  {event.ongoing ? <Badge tone="warn">Ongoing</Badge> : null}
                  {event.source === "manual" ? <Badge tone="info">Staff note</Badge> : null}
                </p>
                {event.detail || event.duration_s ? (
                  <p className="timeline-detail">
                    {event.detail ?? ""}
                    {event.source === "detector" && event.duration_s && !event.detail?.startsWith("Lasted")
                      ? ` ${duration(event.duration_s)}`
                      : ""}
                  </p>
                ) : null}
              </div>
              <span className="timeline-marker" data-tone={event.source === "manual" ? "idle" : SEVERITY_TONE[event.severity]} aria-hidden="true" />
            </li>
          ))}
        </ol>
      ) : (
        <Empty>
          No events in this window. Clinura logs sustained heart-rate and breathing changes, HRV dips,
          movement, eyes-closed rest and time out of view.
        </Empty>
      )}
    </Panel>
  );
}
