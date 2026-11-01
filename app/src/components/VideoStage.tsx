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

// Landmark shown with the live angle label (left, right) per exercise.
const JOINT_INDEX: Record<string, [number, number]> = {
  squat: [25, 26],
  lunge: [25, 26],
  knee_extension: [25, 26],
  wall_sit: [25, 26],
  sit_to_stand: [23, 24],
  glute_bridge: [23, 24],
  marching: [23, 24],
  plank: [23, 24],
  pushup: [13, 14],
  bicep_curl: [13, 14],
  shoulder_press: [13, 14],
  lateral_raise: [11, 12],
  arm_raise: [11, 12],
  jumping_jack: [11, 12],
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
  if (mode === "body") {
    if (metrics.body?.error) return { title: "Body tracking unavailable", copy: metrics.body.error, error: true };
    const handExercise = ["fist_open_close", "pinch", "finger_spread"].includes(metrics.body?.session?.exercise ?? "");
    if (handExercise) {
      return metrics.body?.hand ? null : { title: "Show your hand", copy: "Hold one hand up to the camera, palm facing it." };
    }
    if (metrics.body && !metrics.body.detected) {
      return {
        title: "Step back into view",
        copy: "Stand 2–3 m from the camera so your whole body, head to feet, is visible.",
      };
    }
    return null;
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

    const body = data?.body;
    if (data?.mode === "body" && body?.landmarks) {
      const pts = body.landmarks;
      const accent = cssVar("--roi-palm");
      ctx.lineCap = "round";
      ctx.lineWidth = 4;
      ctx.strokeStyle = hexAlpha(accent, 0.9);
      for (const [a, b] of body.skeleton) {
        const pa = pts[a];
        const pb = pts[b];
        if (!pa || !pb || pa[2] < 0.5 || pb[2] < 0.5) continue;
        ctx.beginPath();
        ctx.moveTo(toX(pa[0]), toY(pa[1]));
        ctx.lineTo(toX(pb[0]), toY(pb[1]));
        ctx.stroke();
      }
      ctx.fillStyle = "#EEF2F5";
      for (const index of [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
        const p = pts[index];
        if (!p || p[2] < 0.5) continue;
        ctx.beginPath();
        ctx.arc(toX(p[0]), toY(p[1]), 4.5, 0, Math.PI * 2);
        ctx.fill();
      }
      const session = body.session;
      if (session?.angle != null && session.active_side) {
        const joint = JOINT_INDEX[session.exercise]?.[session.active_side === "left" ? 0 : 1];
        const p = joint !== undefined ? pts[joint] : undefined;
        if (p && p[2] >= 0.5) {
          const text = `${Math.round(session.angle)}°`;
          ctx.font = "600 15px 'IBM Plex Sans', system-ui, sans-serif";
          const w = ctx.measureText(text).width + 14;
          ctx.fillStyle = "rgba(10, 14, 18, 0.75)";
          ctx.beginPath();
          ctx.roundRect(toX(p[0]) + 10, toY(p[1]) - 12, w, 24, 5);
          ctx.fill();
          ctx.fillStyle = "#EEF2F5";
          ctx.textBaseline = "middle";
          ctx.fillText(text, toX(p[0]) + 17, toY(p[1]));
        }
      }
    }

    if (data?.mode === "body" && body?.hand) {
      const h = body.hand;
      const bones = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
      ctx.strokeStyle = hexAlpha(cssVar("--roi-right"), 0.95);
      ctx.lineWidth = 3;
      for (const [a, b] of bones) {
        ctx.beginPath();
        ctx.moveTo(toX(h[a][0]), toY(h[a][1]));
        ctx.lineTo(toX(h[b][0]), toY(h[b][1]));
        ctx.stroke();
      }
      ctx.fillStyle = "#EEF2F5";
      for (const p of h) {
        ctx.beginPath();
        ctx.arc(toX(p[0]), toY(p[1]), 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
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
    mode === "body"
      ? metrics?.body?.detected
        ? "Body tracked"
        : "Searching for body"
      : mode !== "face"
      ? metrics?.experimental?.active
        ? "Region locked"
        : "Searching"
      : !metrics?.face_detected
        ? "No face"
        : metrics.pose_ok
          ? "Face locked"
          : "Turn toward the camera";
  const lockTone =
    mode === "body"
      ? metrics?.body?.detected
        ? "ok"
        : "warn"
      : mode !== "face"
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
