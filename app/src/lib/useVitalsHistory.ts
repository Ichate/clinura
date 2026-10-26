import { useEffect, useRef, useState } from "react";

import type { Metrics, SignalStatus } from "./types";

export interface HistoryPoint {
  t: number;
  bpm: number | null;
  provisional: boolean;
  confidence: number;
  status: SignalStatus;
  snr: number | null;
}

export function useVitalsHistory(metrics: Metrics | null, seconds = 60): HistoryPoint[] {
  const [points, setPoints] = useState<HistoryPoint[]>([]);
  const lastRef = useRef(0);

  useEffect(() => {
    if (!metrics) return;
    const t = metrics.timestamp;
    if (lastRef.current !== 0 && t - lastRef.current < 0.95) return;
    lastRef.current = t;
    const bpm = metrics.bpm ?? metrics.provisional_bpm;
    const point: HistoryPoint = {
      t,
      bpm,
      provisional: metrics.bpm === null && metrics.provisional_bpm !== null,
      confidence: metrics.confidence,
      status: metrics.status,
      snr: metrics.snr_db,
    };
    setPoints((previous) => {
      const next = [...previous, point];
      return next.filter((item) => item.t >= t - seconds);
    });
  }, [metrics, seconds]);

  return points;
}
