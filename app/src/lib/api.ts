import type {
  ChatMessage,
  ClinicalEvent,
  HandoverResult,
  TelemetrySummary,
} from "./types";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError("Cannot reach the Clinura engine. Check that it is running.", 0);
  }
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = "Check the request and try again.";
    } catch {
      message = response.statusText || message;
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export interface AssistantStatus {
  configured: boolean;
  model: string;
}

export const api = {
  assistant: () => request<AssistantStatus>("/api/assistant"),
  telemetry: (hours: number) => request<TelemetrySummary>(`/api/telemetry?hours=${hours}`),
  events: (hours: number) => request<ClinicalEvent[]>(`/api/events?hours=${hours}`),
  addNote: (title: string, detail?: string) =>
    request<{ ok: boolean; id: number }>("/api/events", {
      method: "POST",
      body: JSON.stringify({ title, detail: detail || null }),
    }),
  handover: (hours: number, bed: string, patient: string) =>
    request<HandoverResult>("/api/handover", {
      method: "POST",
      body: JSON.stringify({ hours, bed: bed || null, patient: patient || null }),
    }),
  ask: (question: string, hours: number, history: ChatMessage[]) =>
    request<{ answer: string; model: string }>("/api/ask", {
      method: "POST",
      body: JSON.stringify({
        question,
        hours,
        history: history
          .filter((message) => !message.error)
          .map(({ role, content }) => ({ role, content })),
      }),
    }),
};
