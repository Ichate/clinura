import type { Tone } from "./types";

export function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

export function num(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "–";
  return value.toFixed(digits);
}

export function snrTone(snr: number | null | undefined): Tone {
  if (snr === null || snr === undefined) return "idle";
  if (snr >= 3) return "ok";
  if (snr >= 0) return "warn";
  return "bad";
}

export function scoreTone(score: number): Tone {
  if (score >= 0.66) return "ok";
  if (score >= 0.33) return "warn";
  return "bad";
}

export function mean(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function cssVar(name: string, fallback = "#888"): string {
  if (typeof window === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export function shortDate(value: string | null | undefined): string {
  if (!value) return "–";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "–"
    : date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function duration(seconds: number | null): string {
  if (seconds === null) return "";
  if (seconds < 90) return `${Math.max(1, Math.round(seconds))} s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}
