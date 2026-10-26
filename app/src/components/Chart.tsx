import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

import { cssVar } from "../lib/format";

interface Props {
  x: number[];
  y: (number | null)[];
  xRange: [number, number];
  color: string;
  height?: number;
  showY?: boolean;
  points?: boolean;
  fill?: boolean;
  xTick?: (value: number) => string;
  yTick?: (value: number) => string;
  hLines?: number[];
  vLines?: number[];
  empty?: string;
  label: string;
}

export function Chart({
  x,
  y,
  xRange,
  color,
  height = 128,
  showY = false,
  points = false,
  fill = false,
  xTick,
  yTick,
  hLines,
  vLines,
  empty = "Waiting for data",
  label,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const propsRef = useRef({ hLines, vLines, xRange, xTick, yTick, color });
  propsRef.current = { hLines, vLines, xRange, xTick, yTick, color };

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const drawGuides = (u: uPlot) => {
      const { ctx } = u;
      const { left, top, width, height: plotHeight } = u.bbox;
      const dpr = window.devicePixelRatio || 1;
      ctx.save();
      ctx.lineWidth = dpr;
      ctx.setLineDash([3 * dpr, 3 * dpr]);
      ctx.strokeStyle = cssVar("--ink-3");
      for (const line of propsRef.current.hLines ?? []) {
        const pos = u.valToPos(line, "y", true);
        if (pos < top || pos > top + plotHeight) continue;
        ctx.beginPath();
        ctx.moveTo(left, pos);
        ctx.lineTo(left + width, pos);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.strokeStyle = cssVar(propsRef.current.color);
      for (const line of propsRef.current.vLines ?? []) {
        const pos = u.valToPos(line, "x", true);
        if (pos < left || pos > left + width) continue;
        ctx.beginPath();
        ctx.moveTo(pos, top);
        ctx.lineTo(pos, top + plotHeight);
        ctx.stroke();
      }
      ctx.restore();
    };

    const axisFont = "500 10px 'IBM Plex Sans', system-ui, sans-serif";
    const plot = new uPlot(
      {
        width: Math.max(1, container.clientWidth),
        height,
        padding: [6, 14, 0, showY ? 0 : 10],
        legend: { show: false },
        cursor: { show: false },
        scales: {
          x: { time: false, auto: false, range: () => propsRef.current.xRange },
          y: {
            auto: true,
            range: (_u, min, max) => {
              const pad = (max - min) * 0.15 || 1;
              return [min - pad, max + pad];
            },
          },
        },
        axes: [
          {
            stroke: () => cssVar("--ink-3"),
            grid: { stroke: () => cssVar("--line"), width: 1 },
            ticks: { show: false },
            font: axisFont,
            size: 24,
            space: 48,
            values: (_u, splits) =>
              splits.map((v) => (propsRef.current.xTick ? propsRef.current.xTick(v) : String(v))),
          },
          showY
            ? {
                stroke: () => cssVar("--ink-3"),
                grid: { show: false },
                ticks: { show: false },
                font: axisFont,
                size: 34,
                values: (_u, splits) =>
                  splits.map((v) =>
                    propsRef.current.yTick ? propsRef.current.yTick(v) : v.toFixed(0),
                  ),
              }
            : { show: false },
        ],
        series: [
          {},
          {
            stroke: () => cssVar(propsRef.current.color),
            width: 1.5,
            fill: fill ? () => `${cssVar(propsRef.current.color)}1f` : undefined,
            fillTo: (u) => u.scales.y.min ?? 0,
            points: {
              show: points,
              size: 4,
              fill: () => cssVar(propsRef.current.color),
              stroke: () => cssVar(propsRef.current.color),
            },
          },
        ],
        hooks: { draw: [drawGuides] },
      },
      [[], []],
      container,
    );
    plotRef.current = plot;
    const observer = new ResizeObserver(() =>
      plot.setSize({ width: Math.max(1, container.clientWidth), height }),
    );
    observer.observe(container);
    return () => {
      observer.disconnect();
      plot.destroy();
      plotRef.current = null;
    };
  }, [height, showY, points, fill]);

  useEffect(() => {
    const plot = plotRef.current;
    if (!plot) return;
    const count = Math.min(x.length, y.length);
    plot.setData([x.slice(0, count), y.slice(0, count)], false);
    plot.setScale("x", { min: xRange[0], max: xRange[1] });
  }, [x, y, xRange[0], xRange[1]]);

  const hasData = x.length > 1 && y.some((value) => value !== null);

  return (
    <div className="chart" style={{ height }} role="img" aria-label={label}>
      <div ref={containerRef} className="chart-canvas" />
      {!hasData ? <div className="chart-empty">{empty}</div> : null}
    </div>
  );
}
