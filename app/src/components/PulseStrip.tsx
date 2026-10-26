import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

import { cssVar } from "../lib/format";
import type { Metrics } from "../lib/types";
import { Segmented } from "./ui";

interface Props {
  metrics: Metrics | null;
  windowSec: number;
  onWindowChange: (seconds: number) => void;
}

const HEIGHT = 176;

function drawPaper(u: uPlot, windowSec: number) {
  const { ctx } = u;
  const { left, top, width, height } = u.bbox;
  const dpr = window.devicePixelRatio || 1;
  ctx.save();
  ctx.fillStyle = cssVar("--paper");
  ctx.fillRect(left, top, width, height);
  const minorStep = windowSec <= 1 ? 0.04 : windowSec <= 5 ? 0.2 : 0.4;
  const pxPerSec = width / windowSec;
  const minorPx = minorStep * pxPerSec;
  const rows = Math.max(4, Math.round(height / minorPx));
  const rowPx = height / rows;
  const minor = cssVar("--grid-minor");
  const major = cssVar("--grid-major");
  ctx.lineWidth = dpr;
  for (let i = 0; i <= windowSec / minorStep + 0.5; i++) {
    const x = left + width - i * minorPx;
    ctx.strokeStyle = i % 5 === 0 ? major : minor;
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, top);
    ctx.lineTo(Math.round(x) + 0.5, top + height);
    ctx.stroke();
  }
  for (let j = 0; j <= rows; j++) {
    const y = top + j * rowPx;
    ctx.strokeStyle = j % 5 === 0 ? major : minor;
    ctx.beginPath();
    ctx.moveTo(left, Math.round(y) + 0.5);
    ctx.lineTo(left + width, Math.round(y) + 0.5);
    ctx.stroke();
  }
  ctx.restore();
}

export function PulseStrip({ metrics, windowSec, onWindowChange }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const beatsRef = useRef<number[]>([]);
  const windowRef = useRef(windowSec);
  windowRef.current = windowSec;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const drawBeats = (u: uPlot) => {
      const { ctx } = u;
      const { left, top, width } = u.bbox;
      const dpr = window.devicePixelRatio || 1;
      ctx.save();
      ctx.fillStyle = cssVar("--hr");
      for (const beat of beatsRef.current) {
        const x = u.valToPos(beat, "x", true);
        if (x < left || x > left + width) continue;
        ctx.beginPath();
        ctx.moveTo(x - 4 * dpr, top);
        ctx.lineTo(x + 4 * dpr, top);
        ctx.lineTo(x, top + 6 * dpr);
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    };
    const plot = new uPlot(
      {
        width: Math.max(1, container.clientWidth),
        height: HEIGHT,
        padding: [0, 14, 0, 14],
        legend: { show: false },
        cursor: { show: false },
        scales: {
          x: { time: false, auto: false, range: () => [-windowRef.current, 0] },
          y: {
            auto: true,
            range: (_u, min, max) => {
              const pad = (max - min) * 0.2 || 1;
              return [min - pad, max + pad];
            },
          },
        },
        axes: [
          {
            stroke: () => cssVar("--ink-3"),
            grid: { show: false },
            ticks: { show: false },
            font: "500 10px 'IBM Plex Sans', system-ui, sans-serif",
            size: 22,
            space: 90,
            incrs: [0.5, 1, 2, 5],
            values: (_u, splits) =>
              splits.map((v) =>
                Math.abs(v) < 1e-6 ? "now" : `${Number.isInteger(v) ? v.toFixed(0) : v.toFixed(2).replace(/0$/, "")} s`,
              ),
          },
          { show: false },
        ],
        series: [{}, { stroke: () => cssVar("--trace"), width: 1.75, points: { show: false } }],
        hooks: {
          drawClear: [(u) => drawPaper(u, windowRef.current)],
          draw: [drawBeats],
        },
      },
      [[], []],
      container,
    );
    plotRef.current = plot;
    const observer = new ResizeObserver(() =>
      plot.setSize({ width: Math.max(1, container.clientWidth), height: HEIGHT }),
    );
    observer.observe(container);
    return () => {
      observer.disconnect();
      plot.destroy();
      plotRef.current = null;
    };
  }, []);

  useEffect(() => {
    const plot = plotRef.current;
    if (!plot) return;
    const x = metrics?.waveform.t ?? [];
    const y = metrics?.waveform.y ?? [];
    const count = Math.min(x.length, y.length);
    const start = Math.max(0, x.findIndex((value) => value >= -windowSec));
    beatsRef.current = metrics?.beats ?? [];
    plot.setData([x.slice(start, count), y.slice(start, count)], false);
    plot.setScale("x", { min: -windowSec, max: 0 });
  }, [metrics, windowSec]);

  const hasSignal = (metrics?.waveform.y.length ?? 0) > 2;

  return (
    <section className="strip" aria-label="Pulse waveform">
      <header className="strip-head">
        <h2 className="panel-title">Pulse waveform</h2>
        <span className="panel-meta">
          {metrics?.fs ? `${metrics.fs.toFixed(0)} Hz, 0.75–2.8 Hz band, beats marked` : "0.75–2.8 Hz band"}
        </span>
        <div className="panel-actions">
          <Segmented
            label="Waveform time window"
            value={windowSec}
            onChange={onWindowChange}
            options={[
              { value: 2, label: "2 s" },
              { value: 5, label: "5 s" },
              { value: 10, label: "10 s" },
            ]}
          />
        </div>
      </header>
      <div className="strip-plot">
        <div ref={containerRef} />
        {!hasSignal ? <div className="chart-empty">The trace appears once a face is tracked</div> : null}
      </div>
    </section>
  );
}
