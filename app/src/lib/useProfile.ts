import { useCallback, useState } from "react";

export type Period = "morning" | "afternoon" | "evening";

export const PERIODS: { value: Period; label: string }[] = [
  { value: "morning", label: "Morning" },
  { value: "afternoon", label: "Afternoon" },
  { value: "evening", label: "Evening" },
];

export function currentPeriod(date = new Date()): Period {
  const hour = date.getHours();
  if (hour >= 5 && hour < 12) return "morning";
  if (hour >= 12 && hour < 18) return "afternoon";
  return "evening";
}

export function periodLabel(period: Period): string {
  return PERIODS.find((item) => item.value === period)?.label ?? "Today";
}

export interface CheckInSummary {
  bpm: number | null;
  rmssd_ms: number | null;
  sdnn_ms: number | null;
  pnn50: number | null;
  respiration_bpm: number | null;
  confidence: number;
  coverage: number;
  duration_s: number;
  created_at: string;
  period: Period;
  lighting_mean: number | null;
  lighting_drift: number | null;
  snr_db: number | null;
  quality: number | null;
}

export interface Profile {
  name: string;
  skinTone: string;
  showProvisional: boolean;
  bed: string;
  patient: string;
  baselines: Record<Period, CheckInSummary | null>;
  updated_at: string;
}

export const SKIN_TONE_UNSET = "Prefer not to say";

const KEY = "clinura.profile.v2";
const LEGACY_KEY = "clinura.baseline.v1";

function emptyProfile(): Profile {
  return {
    name: "",
    skinTone: SKIN_TONE_UNSET,
    showProvisional: true,
    bed: "",
    patient: "",
    baselines: { morning: null, afternoon: null, evening: null },
    updated_at: new Date().toISOString(),
  };
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function normalizeBaseline(value: unknown, fallbackPeriod: Period): CheckInSummary | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const period =
    raw.period === "morning" || raw.period === "afternoon" || raw.period === "evening"
      ? raw.period
      : fallbackPeriod;
  return {
    bpm: numberOrNull(raw.bpm),
    rmssd_ms: numberOrNull(raw.rmssd_ms),
    sdnn_ms: numberOrNull(raw.sdnn_ms),
    pnn50: numberOrNull(raw.pnn50),
    respiration_bpm: numberOrNull(raw.respiration_bpm),
    confidence: numberOrNull(raw.confidence) ?? 0,
    coverage: numberOrNull(raw.coverage) ?? 0,
    duration_s: numberOrNull(raw.duration_s) ?? 60,
    created_at: typeof raw.created_at === "string" ? raw.created_at : new Date().toISOString(),
    period,
    lighting_mean: numberOrNull(raw.lighting_mean),
    lighting_drift: numberOrNull(raw.lighting_drift),
    snr_db: numberOrNull(raw.snr_db),
    quality: numberOrNull(raw.quality),
  };
}

function load(): Profile {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<Profile>;
      const base = emptyProfile();
      return {
        ...base,
        name: typeof parsed.name === "string" ? parsed.name : base.name,
        skinTone: typeof parsed.skinTone === "string" ? parsed.skinTone : base.skinTone,
        showProvisional:
          typeof parsed.showProvisional === "boolean" ? parsed.showProvisional : true,
        bed: typeof parsed.bed === "string" ? parsed.bed : "",
        patient: typeof parsed.patient === "string" ? parsed.patient : "",
        baselines: {
          morning: normalizeBaseline(parsed.baselines?.morning, "morning"),
          afternoon: normalizeBaseline(parsed.baselines?.afternoon, "afternoon"),
          evening: normalizeBaseline(parsed.baselines?.evening, "evening"),
        },
      };
    }
    const legacy = localStorage.getItem(LEGACY_KEY);
    if (legacy) {
      const old = JSON.parse(legacy) as Record<string, unknown>;
      const period = currentPeriod(
        new Date(typeof old.created_at === "string" ? old.created_at : Date.now()),
      );
      const profile = emptyProfile();
      profile.baselines[period] = normalizeBaseline(old, period);
      localStorage.setItem(KEY, JSON.stringify(profile));
      return profile;
    }
  } catch {
    return emptyProfile();
  }
  return emptyProfile();
}

function save(profile: Profile): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(profile));
  } catch {
    return;
  }
}

export function useProfile() {
  const [profile, setProfile] = useState<Profile>(load);

  const update = useCallback((patch: Partial<Profile>) => {
    setProfile((previous) => {
      const next = { ...previous, ...patch, updated_at: new Date().toISOString() };
      save(next);
      return next;
    });
  }, []);

  const saveBaseline = useCallback((summary: CheckInSummary) => {
    setProfile((previous) => {
      const next = {
        ...previous,
        baselines: { ...previous.baselines, [summary.period]: summary },
        updated_at: new Date().toISOString(),
      };
      save(next);
      return next;
    });
  }, []);

  const reset = useCallback(() => {
    const next = emptyProfile();
    save(next);
    try {
      localStorage.removeItem(LEGACY_KEY);
    } catch {
      setProfile(next);
    }
    setProfile(next);
  }, []);

  return { profile, update, saveBaseline, reset };
}
