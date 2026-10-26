import type { ReactNode } from "react";

import { clamp01 } from "../lib/format";
import type { Tone } from "../lib/types";

interface PanelProps {
  title: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
  id?: string;
}

export function Panel({ title, meta, actions, className, children, id }: PanelProps) {
  return (
    <section className={`panel ${className ?? ""}`} aria-labelledby={id}>
      <header className="panel-head">
        <h2 className="panel-title" id={id}>
          {title}
        </h2>
        {meta ? <span className="panel-meta">{meta}</span> : null}
        {actions ? <div className="panel-actions">{actions}</div> : null}
      </header>
      {children}
    </section>
  );
}

export function Badge({ tone = "idle", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span className="badge" data-tone={tone}>
      {children}
    </span>
  );
}

export function Dot({ tone = "idle" }: { tone?: Tone }) {
  return <span className="dot" data-tone={tone} aria-hidden="true" />;
}

export function Meter({
  value,
  tone = "info",
  label,
}: {
  value: number;
  tone?: Tone;
  label?: string;
}) {
  const pct = Math.round(clamp01(value) * 100);
  return (
    <div
      className="meter"
      role="meter"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      aria-label={label}
    >
      <span className="meter-fill" data-tone={tone} style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Stat({
  label,
  value,
  unit,
  tone,
}: {
  label: ReactNode;
  value: ReactNode;
  unit?: string;
  tone?: Tone;
}) {
  return (
    <div className="stat">
      <dt>{label}</dt>
      <dd data-tone={tone}>
        {value}
        {unit ? <span className="unit">{unit}</span> : null}
      </dd>
    </div>
  );
}

export function Stats({ children, columns = 2 }: { children: ReactNode; columns?: number }) {
  return (
    <dl className="stats" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
      {children}
    </dl>
  );
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={String(option.value)}
          type="button"
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Note({ children }: { children: ReactNode }) {
  return <p className="note">{children}</p>;
}
