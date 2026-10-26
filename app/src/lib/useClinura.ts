import { useCallback, useEffect, useRef, useState } from "react";

import type { ConnectionState, Hello, Metrics, MonitorMode } from "./types";

const explicitUrl = import.meta.env.VITE_WS_URL as string | undefined;

function websocketUrl(): string {
  if (explicitUrl) return explicitUrl;
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/ws`;
}

export function useClinura() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [hello, setHello] = useState<Hello | null>(null);
  const [frame, setFrame] = useState<ImageBitmap | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [mode, setModeState] = useState<MonitorMode>("face");
  const socketRef = useRef<WebSocket | null>(null);
  const modeRef = useRef<MonitorMode>("face");
  const retryRef = useRef(0);
  const timerRef = useRef<number | null>(null);
  const decodingRef = useRef(false);

  useEffect(() => {
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      setConnection(retryRef.current === 0 ? "connecting" : "reconnecting");
      const socket = new WebSocket(websocketUrl());
      socket.binaryType = "blob";
      socketRef.current = socket;

      socket.onopen = () => {
        if (disposed) return;
        retryRef.current = 0;
        setConnection("live");
        socket.send(JSON.stringify({ type: "set_mode", mode: modeRef.current }));
      };

      socket.onmessage = (event) => {
        if (typeof event.data === "string") {
          try {
            const message = JSON.parse(event.data) as Hello | Metrics;
            if (message.type === "hello") setHello(message);
            else if (message.type === "metrics") setMetrics(message);
          } catch {
            return;
          }
          return;
        }
        if (event.data instanceof Blob && !decodingRef.current) {
          decodingRef.current = true;
          createImageBitmap(event.data)
            .then((bitmap) => {
              if (disposed) bitmap.close();
              else setFrame(bitmap);
            })
            .catch(() => undefined)
            .finally(() => {
              decodingRef.current = false;
            });
        }
      };

      socket.onclose = () => {
        if (disposed) return;
        setConnection("reconnecting");
        setFrame(null);
        retryRef.current = Math.min(retryRef.current + 1, 6);
        const delay = Math.min(500 * 2 ** retryRef.current, 8000);
        timerRef.current = window.setTimeout(connect, delay);
      };

      socket.onerror = () => socket.close();
    };

    connect();
    return () => {
      disposed = true;
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, []);

  const send = useCallback((payload: object) => {
    const socket = socketRef.current;
    if (socket !== null && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(payload));
    }
  }, []);

  const setMode = useCallback(
    (nextMode: MonitorMode) => {
      modeRef.current = nextMode;
      setModeState(nextMode);
      send({ type: "set_mode", mode: nextMode });
    },
    [send],
  );

  const resetSignal = useCallback(() => send({ type: "reset" }), [send]);

  return { metrics, hello, frame, connection, mode, setMode, resetSignal };
}
