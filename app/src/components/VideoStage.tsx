import { useCallback, useEffect, useRef } from "react";

import { cssVar } from "../lib/format";
import type { ConnectionState, Metrics, MonitorMode } from "../lib/types";

interface Props {
  frame: ImageBitmap | null;
  metrics: Metrics | null;
  connection: ConnectionState;
  mode: MonitorMode;
}

const ROI_TOKENS: Record<string, string> = {
  forehead: "--roi-forehead",
  right_cheek: "--roi-right",
  left_cheek: "--roi-left",
};

const ROI_LABELS: Record<string, string> = {
  forehead: "Forehead",
  right_cheek: "Right cheek",
  left_cheek: "Left cheek",
};

function hexAlpha(hex: string, alpha: number): string {
  const value = hex.replace("#", "");
  if (value.length !== 6) return hex;
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function guide(metrics: Metrics | null, connection: ConnectionState, mode: MonitorMode) {
  if (connection !== "live") {
    return {
      title: "Connecting to the engine",
      copy: "Start Clinura with python main.py. This view reconnects on its own.",
    };
  }
  if (!metrics || metrics.engine_state === "starting") {
    return { title: "Starting the camera", copy: "This takes a few seconds." };
  }
  if (metrics.error) {
    return { title: "Camera unavailable", copy: metrics.error, error: true };
  }
  if (mode !== "face") {
    if (metrics.experimental?.active) return null;
    return {
      title: mode === "tongue" ? "Show your tongue" : "Show your palm",
      copy:
        metrics.experimental?.message ??
        (mode === "tongue"
          ? "Open your mouth and hold your tongue toward the camera."
          : "Hold your palm toward the camera in even light."),
    };
  }
  if (!metrics.face_detected) {
    return {
      title: "Position your face in the outline",
      copy: "Sit about an arm's length away, facing soft, even light.",
    };
  }
  return null;
}

export function VideoStage({ frame, metrics, connection, mode }: Props) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bitmapRef = useRef<ImageBitmap | null>(null);
  const metricsRef = useRef<Metrics | null>(metrics);
  metricsRef.current = metrics;

  const draw = useCallback(() => {
    const wrapper = wrapperRef.current;
    const canvas = canvasRef.current;
    if (!wrapper || !canvas) return;
    const rect = wrapper.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const bitmap = bitmapRef.current;
    const data = metricsRef.current;
    let dx = 0;
    let dy = 0;
    let dw = width;
    let dh = height;
    if (bitmap) {
      const scale = Math.min(width / bitmap.width, height / bitmap.height);
      dw = bitmap.width * scale;
      dh = bitmap.height * scale;
      dx = (width - dw) / 2;
      dy = (height - dh) / 2;
      ctx.drawImage(bitmap, dx, dy, dw, dh);
    } else {
      dw = Math.min(width, (height * 16) / 9);
      dh = (dw * 9) / 16;
      dx = (width - dw) / 2;
      dy = (height - dh) / 2;
    }
    const toX = (value: number) => dx + value * dw;
    const toY = (value: number) => dy + value * dh;

    const polygon = (points: [number, number][]) => {
      ctx.beginPath();
      points.forEach(([x, y], index) => {
        if (index === 0) ctx.moveTo(toX(x), toY(y));
        else ctx.lineTo(toX(x), toY(y));
      });
      ctx.closePath();
    };

    const label = (points: [number, number][], text: string, color: string) => {
      const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
      const top = Math.min(...points.map((p) => p[1]));
      ctx.font = "500 11px 'IBM Plex Sans', system-ui, sans-serif";
      const textWidth = ctx.measureText(text).width;
      const w = textWidth + 18;
      const h = 20;
      const px = Math.max(dx + 4, Math.min(dx + dw - w - 4, toX(cx) - w / 2));
      const py = Math.max(dy + 4, toY(top) - h - 6);
      ctx.fillStyle = "rgba(10, 14, 18, 0.72)";
      ctx.beginPath();
      ctx.roundRect(px, py, w, h, 4);
      ctx.fill();
      ctx.fillStyle = color;
      ctx.fillRect(px + 7, py + 7, 3, 6);
      ctx.fillStyle = "#EEF2F5";
      ctx.textBaseline = "middle";
      ctx.fillText(text, px + 14, py + h / 2 + 0.5);
    };

    if (data && data.mode !== "palm" && data.face_detected) {
      if (data.face_contour && data.face_contour.length > 2) {
        polygon(data.face_contour);
        ctx.strokeStyle = "rgba(238, 242, 245, 0.35)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      for (const roi of data.rois) {
        if (roi.contour.length < 3 || roi.pixels <= 0) continue;
        const color = cssVar(ROI_TOKENS[roi.name] ?? "--roi-forehead");
        polygon(roi.contour);
        ctx.fillStyle = hexAlpha(color, 0.16);
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        label(roi.contour, ROI_LABELS[roi.name] ?? roi.name, color);
      }
    }

    const experimental = data?.experimental;
    if (experimental?.active && experimental.roi && experimental.roi.contour.length >= 3) {
      const color = cssVar(data?.mode === "tongue" ? "--roi-tongue" : "--roi-palm");
      polygon(experimental.roi.contour);
      ctx.fillStyle = hexAlpha(color, 0.18);
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      const text =
        data?.mode === "tongue"
          ? "Tongue"
          : `Palm${experimental.handedness ? `, ${experimental.handedness.toLowerCase()}` : ""}`;
      label(experimental.roi.contour, text, color);
    }

    if (data && !data.face_detected && data.mode === "face" && !data.error) {
      const gw = Math.min(dw * 0.28, 240);
      const gh = gw * 1.3;
      ctx.save();
      ctx.setLineDash([5, 6]);
      ctx.strokeStyle = "rgba(238, 242, 245, 0.5)";
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.ellipse(dx + dw / 2, dy + dh * 0.46, gw / 2, gh / 2, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }, []);

  useEffect(() => {
    const previous = bitmapRef.current;
    bitmapRef.current = frame;
    draw();
    if (previous && previous !== frame) previous.close();
  }, [frame, draw]);

  useEffect(() => {
    draw();
  }, [metrics, draw]);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, [draw]);

  const notice = guide(metrics, connection, mode);
  const camera = metrics?.camera;
  const lock =
    mode !== "face"
      ? metrics?.experimental?.active
        ? "Region locked"
        : "Searching"
      : !metrics?.face_detected
        ? "No face"
        : metrics.pose_ok
          ? "Face locked"
          : "Turn toward the camera";
  const lockTone =
    mode !== "face"
      ? metrics?.experimental?.active
        ? "ok"
        : "warn"
      : !metrics?.face_detected
        ? "bad"
        : metrics.pose_ok
          ? "ok"
          : "warn";

  return (
    <div className="stage" ref={wrapperRef}>
      <canvas ref={canvasRef} className="stage-canvas" />
      <div className="stage-bar stage-bar--top">
        <span className="stage-chip">
          <span className="dot" data-tone={connection === "live" && frame ? "ok" : "idle"} />
          {connection === "live" && frame ? "Live" : "No video"}
        </span>
        {camera && camera.width > 0 ? (
          <span className="stage-chip">
            {camera.index !== undefined ? `Camera ${camera.index}, ` : ""}
            {camera.width}×{camera.height}, {camera.fps.toFixed(0)} fps
          </span>
        ) : null}
        <span className="stage-chip stage-chip--end">
          <span className="dot" data-tone={connection === "live" ? lockTone : "idle"} />
          {connection === "live" ? lock : "Offline"}
        </span>
      </div>
      {notice ? (
        <div className="stage-notice" data-error={notice.error ? "true" : undefined}>
          <p className="stage-notice-title">{notice.title}</p>
          <p className="stage-notice-copy">{notice.copy}</p>
        </div>
      ) : metrics?.message && mode === "face" ? (
        <div className="stage-bar stage-bar--bottom">
          <span className="stage-chip">{metrics.message}</span>
        </div>
      ) : null}
    </div>
  );
}
